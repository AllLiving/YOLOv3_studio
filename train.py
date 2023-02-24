# from utils import TrainDataset, setup_metrics_for_loader
# from YOLOv3 import YOLOv3
############# 这段代码在本地机器上运行请慎重，容易造成死机#######################
import time
import os
import paddle
import numpy as np

ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
IGNORE_THRESH = .7
NUM_CLASSES = 7
USE_GPU = True

NMS_TOPK = 40
NMS_POSK = 10
NMS_THRESH = 0.45
VALID_THRESH = 0.01

# 表示在 10000步以内步长base_lr; [10000,20000)内步长衰减一次; 高于20000次衰减平方次；
def get_lr(base_lr = 0.0001, lr_decay = 0.1):
    bd = [10000, 20000]
    lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(boundaries=bd, values=lr)
    return learning_rate

if __name__ == '__main__':
    paddle.device.set_device("gpu:0")

    # 由传入参数得到 dataset
    TRAINDIR = '/home/aistudio/work/insects/train'
    TESTDIR = '/home/aistudio/work/insects/test'
    VALIDDIR = '/home/aistudio/work/insects/val'
    train_dataset = TrainDataset(VALIDDIR, mode='train')
    valid_dataset = TrainDataset(VALIDDIR, mode='valid')
    test_dataset = TrainDataset(VALIDDIR, mode='valid')
    # 使用paddle.io.DataLoader创建数据读取器，并设置batchsize，进程数量num_workers等参数
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=2, drop_last=True, use_shared_memory=False)
    valid_loader = paddle.io.DataLoader(valid_dataset, batch_size=5, 
        shuffle=False, num_workers=1, drop_last=False, use_shared_memory=False)
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=0, drop_last=False, use_shared_memory=False)

    # 运行模型开始训练
    model = YOLOv3(num_classes = NUM_CLASSES)
    learning_rate = get_lr()
    opt = paddle.optimizer.Momentum(
                 learning_rate=learning_rate,
                 momentum=0.9,
                 weight_decay=paddle.regularizer.L2Decay(0.0005),
                 parameters=model.parameters())  #创建优化器
    # opt = paddle.optimizer.Adam(learning_rate=learning_rate, weight_decay=paddle.regularizer.L2Decay(0.0005), parameters=model.parameters())

    """
    TODO: 
    1. 记录每个epoch用时 iter_tic = time.time()
    2. epoch前 调用_flops计算
    3. 每个epoch训练完成后 调用eval函数_eval_with_loader
    4. 添加pred循环
    """
    MAX_EPOCH = 1 #200
    # model.train()
    # for epoch in range(MAX_EPOCH):
    #     print('[TRAIN]epoch {}'.format(epoch))
    #     for i, data in enumerate(train_loader()):
    #         img, gt_boxes, gt_labels, img_scale = data
    #         img = paddle.to_tensor(img)
    #         gt_boxes = paddle.to_tensor(gt_boxes)
    #         gt_labels = paddle.to_tensor(gt_labels)
    #         gt_scores = np.ones(gt_labels.shape).astype('float32')
    #         gt_scores = paddle.to_tensor(gt_scores)

    #         print("begin forwarding...")
    #         outputs = model(img)  # 前向传播，输出[P0, P1, P2]
    #         loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
    #                               anchors = ANCHORS,
    #                               anchor_masks = ANCHOR_MASKS,
    #                               ignore_thresh=IGNORE_THRESH,
    #                               use_label_smooth=False)        # 计算损失函数
    #         # print("loss:{}".format(loss))

    #         loss.backward()    # 反向传播计算梯度
    #         opt.step()  # 更新参数
    #         opt.clear_grad()
    #         if i % 2 == 0:
    #             timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
    #             print('{}[TRAIN]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))

    #     # save params of model
    #     if (epoch % 5 == 0) or (epoch == MAX_EPOCH -1):
    #         paddle.save(model.state_dict(), 'yolo_epoch{}'.format(epoch))

        # # 每个epoch结束之后在验证集上进行测试
        # model.eval()
        # for i, data in enumerate(valid_loader()):
        #     img, gt_boxes, gt_labels, img_scale = data
        #     gt_scores = np.ones(gt_labels.shape).astype('float32')
        #     gt_scores = paddle.to_tensor(gt_scores)
        #     img = paddle.to_tensor(img)
        #     gt_boxes = paddle.to_tensor(gt_boxes)
        #     gt_labels = paddle.to_tensor(gt_labels)
        #     outputs = model(img)
        #     loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
        #                           anchors = ANCHORS,
        #                           anchor_masks = ANCHOR_MASKS,
        #                           ignore_thresh=IGNORE_THRESH,
        #                           use_label_smooth=False)
        #     if i % 1 == 0:
        #         timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        #         print('{}[VALID]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))
        # model.train()

    # 训练完成后 在测试集上测试
    """
    outputs[0]_shape:[10, 36, 20, 20]
    gt_boxes_shape: [10, 50, 4]
    gt_labels_shape: [10, 50]
    gt_scores_shape: [10, 50]
    """
    model.set_state_dict(paddle.load("yolo_epoch4"))
    metrics = setup_metrics_for_loader(model)
    for i, data in enumerate(valid_loader()):
        img, gt_boxes, gt_labels, img_scale = data
        img = paddle.to_tensor(img)
        gt_scores = np.ones(gt_labels.shape).astype('float32')
        gt_scores = paddle.to_tensor(gt_scores)
        gt_boxes = paddle.to_tensor(gt_boxes)
        gt_labels = paddle.to_tensor(gt_labels)
        # print("data prepared done.")
        model.eval()
        outputs = model(img)

        # print("len outputs:{}".format(len(outputs)))
        # crt_out = np.asarray(outputs[0])
        # print("shape:{}".format(crt_out.shape))
        # print("gt_boxes_shape:{}".format(gt_boxes.shape))
        # print("gt_labels_shape:{}".format(gt_labels.shape))
        # print("gt_scores_shape:{}".format(gt_scores.shape))

        img_scale = paddle.to_tensor(img_scale).astype('int32')
        bboxes, scores = model.get_pred(outputs,
                                 im_shape=img_scale)
        bboxes_data = bboxes.numpy()
        scores_data = scores.numpy()
        # bboxes_data = bboxes_data[:, :50, :]
        # scores_data = scores_data[:, :, :50]
        # print("bboxes_shape:{}, scores_shape:{}".format(bboxes_data.shape, scores_data.shape))

        """
        TODO： 添加metric的update
        添加 nms 处理预测结果 获得更准确的label和score 以用于metric计算
            nms返回值result如何使用？
        分离labels， bbox 和 scores信息
            检查 json格式信息的最优化导出方法
        result_j_shape: [35~44,6] each item has [labels,score, bbox]
        """
        # get pred labels and scores
        result = multiclass_nms(bboxes_data, scores_data, nms_thresh=0.25)
        total_results = []
        pred_labels = []
        pred_scores = []
        pred_boxes = []
        for j in range(len(result)):
            result_j = result[j]
            img_name_j = "pic_"+str(j)
            total_results.append([img_name_j, result_j.tolist()])
            print("result_j_shape:{}".format(np.asarray(result_j).shape))
        print('processed {} pictures'.format(len(total_results)))
        print("result shape:{}".format(np.asarray(result[0]).shape))
        json.dump(total_results, open('pred_results.json', 'w'))

        print("gt_labels:{}".format(gt_labels))
        # print("gt_scores:{}".format(gt_scores))

        break

        # print(gt_labels[0])
        # print(gt_boxes)
        # for _m in metrics:
        #     _m.update(gt_boxes, gt_labels, img_scale, bboxes_data, scores_data)
        loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                anchors = ANCHORS,
                                anchor_masks = ANCHOR_MASKS,
                                ignore_thresh=IGNORE_THRESH,
                                use_label_smooth=False)
        if i % 2 == 0:
            timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
            print('{}[TEST]iter {}, output loss: {}'.format(timestring, i, loss.numpy()))
        if i > 10:
            break

    # # 累计计算 metrics
    # for _m in metrics:
    #     _m.accumulate()
    #     _m.log()
    #     _m.reset()