import paddle
import numpy as np
import os
import sys
import json
# from backbone import DarkNet53_conv_body,YoloDetectionBlock,ConvBNLayer
# from map_utils import DetectionMAP,prune_zero_padding
# from utils import setup_logger
logger = setup_logger(__name__, output="log/log.txt")

global OUTPUT_DIR, DATASET_PREFIX, TRAIN_DATASETS, INSECT_NAMES
OUTPUT_DIR = "/home/aistudio/log" # "/root/paddlejob/workspace/output/"
DATASET_PREFIX = '/root/paddlejob/workspace/train_data/datasets/'
TRAIN_DATASETS = DATASET_PREFIX+"/data"
INSECT_NAMES = ['Boerner', 'Leconte', 'Linnaeus', 
                'acuminatus', 'armandi', 'coleoptera', 'linnaeus']
"""
上采样函数
为传入特征图以插值法宽高扩大至2倍 其中间变量不参与迭代更新
"""
class Upsample(paddle.nn.Layer):
    def __init__(self, scale=2):
        super(Upsample,self).__init__()
        self.scale = scale

    def forward(self, inputs):
        # get dynamic upsample output shape
        shape_nchw = paddle.shape(inputs)
        shape_hw = paddle.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = paddle.cast(shape_hw, dtype='int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = paddle.nn.functional.interpolate(
            x=inputs, scale_factor=self.scale, mode="NEAREST")
        return out

class Metric(paddle.metric.Metric):
    def name(self):
        return self.__class__.__name__

    def reset(self):
        pass

    def accumulate(self):
        pass

    # paddle.metric.Metric defined :metch:`update`, :meth:`accumulate`
    # :metch:`reset`, in ppdet, we also need following 2 methods:

    # abstract method for logging metric results
    def log(self):
        pass

    # abstract method for getting metric results
    def get_results(self):
        pass

class VOCMetric(Metric):
    def __init__(self,
                 label_list,
                 class_num=20,
                 overlap_thresh=0.5,
                 map_type='11point',
                 is_bbox_normalized=False,
                 evaluate_difficult=False,
                 classwise=False,
                 output_eval=None,
                 save_prediction_only=False):
        # assert os.path.isfile(label_list), \
        #         "label_list {} not a file".format(label_list)

        self.catid2name = {i:name for i,name in enumerate(label_list)}
        # self.clsid2catid, self.catid2name = get_categories('VOC', label_list)

        self.overlap_thresh = overlap_thresh
        self.map_type = map_type
        self.evaluate_difficult = evaluate_difficult
        self.output_eval = output_eval
        self.save_prediction_only = save_prediction_only
        self.detection_map = DetectionMAP(
            class_num=class_num,
            overlap_thresh=overlap_thresh,
            map_type=map_type,
            is_bbox_normalized=is_bbox_normalized,
            evaluate_difficult=evaluate_difficult,
            catid2name=self.catid2name,
            classwise=classwise)

        self.reset()

    def reset(self):
        self.results = {'bbox': [], 'score': [], 'label': []}
        self.detection_map.reset()

    def update(self, gt_boxes, gt_labels, im_shape, bboxes, scores, labels=None, num_anchors = 3, num_classes = 7):
        """
        outputs： YOLOv3输出的预测特征值 
            预测得特征图 list类型 [3, batch_num, anchor_num*(class_num+5), W, H]
        inputs：  真实信息

        TODO:  获得同一批的 预测框、得分和label信息
        1. 传入bboxes尺寸和格式如何？
        2. 每次传入一个 batch的数据；
        3. gt_boxes_shape:[10,50,4]; gt_labels_shape/scores:[10,50];
        4. 传入 bbox归一化
        5. 该文件正常运行即可
        """
        scores = np.transpose(scores, (0,2,1))
        # print("bboxes_shape:{}".format(bboxes.shape))
        # print("scores_shape:{}".format(scores.shape))

        difficult = np.zeros(gt_labels.shape).astype('float32')
        for j in range(len(gt_boxes)):
            gt_box = gt_boxes[j].numpy() if isinstance(
                gt_boxes[j], paddle.Tensor) else gt_boxes[j]
            h, w = im_shape[j]
            # print("im_shape:{},gt_box_shape:{}".format(im_shape[j], gt_boxes[j].shape))
            gt_box = gt_box.reshape((gt_box.shape[0], gt_box.shape[1], 1))
            gt_box = gt_box / np.array([w, h, w, h])
            gt_label = gt_labels[j].numpy() if isinstance(
                gt_labels[j], paddle.Tensor) else gt_labels[j]
            bbox = bboxes[j]
            score = scores[j]

            gt_box, gt_label, difficult = prune_zero_padding(gt_box, gt_label,
                                                                difficult)
            # print("cnt:{},map updating...".format(j))
            # print("gt_box_shape:{}".format(gt_box.shape)) # (50,4,1)
            # print("gt_label_shape:{}".format(gt_label.shape)) # (50,)
            pred_label = np.zeros((50,))
            self.detection_map.update(bbox, score, gt_label, gt_box, gt_label)
        
        # for i,out in enumerate(outputs):
        #     out = paddle.to_tensor(out)
        #     # reshape  yolo_box info  label格式？
        #     reshaped_output = paddle.reshape(out, [-1, num_anchors, num_classes+5, out.shape[2], out.shape[3]])
        #     # obj info + location + classification  vs  bbox scores label
        #     pred_label = reshaped_output[:, :, 5:5+num_classes, :, :].numpy()
        #     pred_bbox = reshaped_output[:, :, 0:4, :, :].numpy()
        #     pred_scores = reshaped_output[:, :, 4, :, :].numpy()

        #     """
        #     TODO:  获得同一批的 预测框、得分和label信息
        #     1. 传入bboxes尺寸和格式如何？
        #     2. 每次传入一个 batch的数据；
        #     3. gt_boxes_shape:[10,50,4]; gt_labels_shape/scores:[10,50];
        #     4. 传入 bbox归一化
        #     """
        #     difficult = np.zeros(gt_labels.shape).astype('float32')
        #     for j in range(len(gt_boxes)):
        #         gt_box = gt_boxes[j].numpy() if isinstance(
        #             gt_boxes[j], paddle.Tensor) else gt_boxes[j]
        #         h, w = im_shape[j]
        #         # print("im_shape:{},gt_box_shape:{}".format(im_shape[j], gt_boxes[j].shape))
        #         gt_box = gt_box.reshape((gt_box.shape[0], gt_box.shape[1], 1))
        #         gt_box = gt_box / np.array([w, h, w, h])
        #         gt_label = gt_labels[j].numpy() if isinstance(
        #             gt_labels[j], paddle.Tensor) else gt_labels[j]

        #         gt_box, gt_label, difficult = prune_zero_padding(gt_box, gt_label,
        #                                                          difficult)
        #         print("P:{},cnt:{},map updating...".format(i, j))
        #         # print("gt_box_shape:{}".format(gt_box.shape)) # (50,4,1)
        #         # print("gt_label_shape:{}".format(gt_label.shape)) # (50,)
        #         pred_label = np.zeros((50,))
        #         self.detection_map.update(pred_bbox, pred_scores, pred_label, gt_box, gt_label)
        # print("Finish update.")

    def accumulate(self):
        output = "bbox.json"
        if self.output_eval:
            output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results, f)
                logger.info('The bbox result is saved to bbox.json.')
        if self.save_prediction_only:
            return

        logger.info("Accumulating evaluatation results...")
        self.detection_map.accumulate()

    def log(self):
        map_stat = 100. * self.detection_map.get_map()
        print("[log]: logging...mAP:{}".format(map_stat))
        logger.info("mAP({:.2f}, {}) = {:.2f}%".format(self.overlap_thresh,
                                                       self.map_type, map_stat))

    def get_results(self):
        return {'bbox': [self.detection_map.get_map()]}


"""
该类用于构建YOLOv3的网络和损失函数
  使用不同卷积程度的特征图以实现多尺度检测
过程描述：
    - DarkNet53_conv_body返回其特征提取的 C0，C1和C2
    - YoloDetectionBlk特征提取后得到route和tip
        其传入参数为 修改后的r0张量concat张量C1 丰富语义时特征图像素点多
        此处 route_blk_2用于修改r0使其通道数减半
    - Conv2D卷积tip张量 方能得到预测张量P0
    1. 其中route主要用于concat环节构建残差块
    2. tip用于后续特征提取生成预测张量P
    3. C张量来自骨干网络的特征提取和输出
"""
class YOLOv3(paddle.nn.Layer):
    def __init__(self, num_classes=7):
        super(YOLOv3,self).__init__()

        self.num_classes = num_classes
        num_filters = 3 * (self.num_classes + 5)
        self.block = DarkNet53_conv_body()
        self.block_outputs = []
        self.yolo_blocks = []
        self.route_blocks_2 = []
        self._init_metrics()
        # 构造YOLOv3中的关键层
        for i in range(3):
            # yolo_blocks序列 由Ci生成route和tip特征图 通道数确定 (ci生成ri和ti)
            yolo_block = self.add_sublayer(
                "yolo_detecton_block_%d" % (i),
                YoloDetectionBlock(
                                   ch_in=512//(2**i)*2 if i==0 else 512//(2**i)*2 + 512//(2**i),
                                   ch_out = 512//(2**i)))
            self.yolo_blocks.append(yolo_block)

            # 承接yolo_detect(通道数同) 实现从tip生成P0 (ti生成pi)
            block_out = self.add_sublayer(
                "block_out_%d" % (i),
                paddle.nn.Conv2D(in_channels=512//(2**i)*2,
                       out_channels=num_filters,
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       weight_attr=paddle.ParamAttr(
                           initializer=paddle.nn.initializer.Normal(0., 0.02)),
                       bias_attr=paddle.ParamAttr(
                           initializer=paddle.nn.initializer.Constant(0.0),
                           regularizer=paddle.regularizer.L2Decay(0.))))
            self.block_outputs.append(block_out)

            if i < 2:
                # 对ri进行卷积 减半其通道数以便后续concat
                route = self.add_sublayer("route2_%d"%i,
                                          ConvBNLayer(ch_in=512//(2**i),
                                                      ch_out=256//(2**i),
                                                      kernel_size=1,
                                                      stride=1,
                                                      padding=0))
                self.route_blocks_2.append(route)
            # 将ri放大以便跟c_{i+1}保持同样的尺寸
            self.upsample = Upsample()

    def _init_metrics(self, validate=False):
        self._metrics = [
            VOCMetric(
                label_list=INSECT_NAMES,
                class_num=self.num_classes,
                classwise=False,
                output_eval=OUTPUT_DIR)
        ]
    
    # 由以上描述分析C张量输出其各预测层 Pi
    def forward(self, inputs):
        outputs = []
        route = []
        # print("inputs_shape:{}".format(inputs.shape))
        blocks = self.block(inputs)
        # 依次处理各C张量
        for i, block in enumerate(blocks):
            # print("dealing with P_{}".format(i))
            if i > 0:
                # 将r_{i-1}经过卷积和上采样之后得到特征图，与这一级的ci进行拼接
                block = paddle.concat([route, block], axis=1)
            # 从ci生成ti和ri 从ti生成pi 将pi放入列表
            route, tip = self.yolo_blocks[i](block)
            block_out = self.block_outputs[i](tip)
            # print("[YOLOv3] cnt:{}, shape:{}".format(i, np.asarray(block_out).shape))
            outputs.append(block_out)

            if i < 2:
                # 对ri进行卷积调整通道数
                route = self.route_blocks_2[i](route)
                # 对ri进行放大，使其尺寸和c_{i+1}保持一致
                route = self.upsample(route)
        # print("YOLO_v3 forwarding done.")
        return outputs

    # outputs:预测特征图Pi;  gt信息:真实特征; 
    """
    outputs:预测特征图Pi;  gt信息:真实特征; 
    yolo_loss由真实框计算其所处网格的三个锚框的IoU 
        选最大者obj=1对其计算三类损失 余者为0 
        选过阈值但非最大IoU的锚框 设obj=-1(忽略)仅计算obj损失
    """
    def get_loss(self, outputs, gtbox, gtlabel, gtscore=None,
                 anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                 anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 ignore_thresh=0.7,
                 use_label_smooth=False):
        """
        使用paddle.vision.ops.yolo_loss，直接计算损失函数，过程更简洁，速度也更快
        """
        self.losses = []
        downsample = 32
        for i, out in enumerate(outputs): # 对三个层级分别求损失函数
            anchor_mask_i = anchor_masks[i]
            loss = paddle.vision.ops.yolo_loss(
                    x=out,  # out是P0, P1, P2中的一个
                    gt_box=gtbox,  # 真实框坐标
                    gt_label=gtlabel,  # 真实框类别
                    gt_score=gtscore,  # 真实框得分，使用mixup训练技巧时需要，不使用该技巧时直接设置为1，形状与gtlabel相同
                    anchors=anchors,   # 锚框尺寸，包含[w0, h0, w1, h1, ..., w8, h8]共9个锚框的尺寸
                    anchor_mask=anchor_mask_i, # 筛选锚框的mask，例如anchor_mask_i=[3, 4, 5]，将anchors中第3、4、5个锚框挑选出来给该层级使用
                    class_num=self.num_classes, # 分类类别数
                    ignore_thresh=ignore_thresh, # 当预测框与真实框IoU > ignore_thresh，标注objectness = -1
                    downsample_ratio=downsample, # 特征图相对于原图缩小的倍数，例如P0是32， P1是16，P2是8
                    use_label_smooth=False)      # 使用label_smooth训练技巧时会用到，这里没用此技巧，直接设置为False
            self.losses.append(paddle.mean(loss))  #mean对每张图片求和
            downsample = downsample // 2 # 下一级特征图的缩放倍数会减半
        return sum(self.losses) # 对每个层级求和

    """
    应用yolo_bx接口由P生成预测框 output为YOLOv3网络输出P张量
    返回三维张量
        total_boxes格式[N,3*C,4]  total_scores格式[N,3*C,num_classes]
        中间变量boxes格式[N,C=K*(5+num_classes),4]
        中间变量scores格式[N,C,class_num]
    其中 函数yolo_bx 处理基于网络输出结果生成预测框 预测结果未必简洁准确 
       可能多个网格的锚框均被采用 因此欲绘制须筛除冗余框
    """
    def get_pred(self,
                 outputs,
                 im_shape=None,
                 anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                 anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 valid_thresh = 0.01):
        downsample = 32
        total_boxes = []
        total_scores = []

        arr_out_0 = np.asarray(outputs[0])
        arr_out_1 = np.asarray(outputs[1])
        arr_out_2 = np.asarray(outputs[2])
        # print("shape:{}".format(arr_out_0.shape))
        # print("shape:{}".format(arr_out_1.shape))
        # print("shape:{}".format(arr_out_2.shape))

        # 遍历 3个P张量 依次得其预测框和预测分
        for i, out in enumerate(outputs):
            anchor_mask = anchor_masks[i]
            anchors_this_level = []
            for m in anchor_mask:
                anchors_this_level.append(anchors[2 * m])
                anchors_this_level.append(anchors[2 * m + 1])

            if isinstance(out, paddle.Tensor):
                pass
            else:
                out = paddle.to_tensor(out)

            # boxes格式[N,C,4]  scores格式[N,C,class_num]
            boxes, scores = paddle.vision.ops.yolo_box(
                   x=out,
                   img_size=im_shape,
                   anchors=anchors_this_level,
                   class_num=self.num_classes,
                   conf_thresh=valid_thresh,
                   downsample_ratio=downsample,
                   name="yolo_box" + str(i))
            total_boxes.append(boxes)
            total_scores.append(
                        paddle.transpose(
                        scores, perm=[0, 2, 1]))
            downsample = downsample // 2

        yolo_boxes = paddle.concat(total_boxes, axis=1)
        yolo_scores = paddle.concat(total_scores, axis=2)
        return yolo_boxes, yolo_scores