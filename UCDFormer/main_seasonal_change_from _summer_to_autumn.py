import torch
import argparse
import datetime
import os
import parser
from fcmeans import FCM
from torch.optim import lr_scheduler

import numpy as np
from saturateSomePercentile import saturateImage
import tensorboardX
import torch
import torchvision
from torch.utils import data
from PIL import Image
from skimage import filters
from skimage import morphology
import math
import torchvision.transforms as transform
from random import shuffle

import img_augm
import dataset
from model import Grid
from utils import prepare_sub_folder, write_loss, write_images, denormalize_vgg_adain, denormalize_vgg, put_tensor_cuda
import torch.nn.functional as F
import sklearn.metrics as mt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral
from pydensecrf.utils import create_pairwise_gaussian
from sklearn.ensemble import RandomForestClassifier

from vgg import VGG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gf_dim', type=int, default=64)
    parser.add_argument('--df_dim', type=int, default=64)
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--init', type=str, default='kaiming')
    parser.add_argument('--path', type=str, default='Path/models/vgg16-397923af.pth')
    parser.add_argument('--tb_file', type=str, default='Path/tensorboard')
    parser.add_argument('--sub_file', type=str, default='new')
    # data
    parser.add_argument('--image_size', type=int, default=256)
    # ouptut
    parser.add_argument('--output_path', type=str, default='Path/output', help='outputs path')
    # train
    parser.add_argument('--epoch', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_policy', type=str, default='constant', help='step/constant')
    parser.add_argument('--step_size', type=int, default=200000)
    parser.add_argument('--gamma', type=float, default=0.5, help='How much to decay learning rate')
    parser.add_argument('--update_D', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=50000)

    # loss weight
    parser.add_argument('--clw', type=float, default=1000, help='content_weight')
    parser.add_argument('--slw', type=float, default=1, help='style_weight')
    parser.add_argument('--tvw', type=float, default=0, help='tv_loss_weight')

    # bilateral grid
    parser.add_argument('--luma_bins', type=int, default=4)
    parser.add_argument('--channel_multiplier', type=int, default=1)
    parser.add_argument('--spatial_bin', type=int, default=16)
    parser.add_argument('--n_input_channel', type=int, default=256)
    parser.add_argument('--n_input_size', type=int, default=64)
    parser.add_argument('--group_num', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--selection', type=str, default='Ax+b')
    parser.add_argument('--inter_selection', type=str, default='A1x+b2')

    # seed
    parser.add_argument('--seed', type=int, default=123)
    opts = parser.parse_args()

    # fix the seed
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    if not os.path.exists(opts.tb_file):
        os.mkdir(opts.tb_file)

    tb_file = opts.tb_file
    sub_file = opts.sub_file
    # Setup logger and output folders
    if not os.path.exists(opts.output_path):
        os.mkdir(opts.output_path)
    output_directory = opts.output_path + sub_file
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    print(checkpoint_directory)
    train_writer = tensorboardX.SummaryWriter(tb_file+sub_file)

    gpu_num = torch.cuda.device_count()

    # prepare dataset
    dataPath_A = "Path/Seasonal_Changes/1_A.bmp"
    dataPath_B = "Path/Seasonal_Changes/1_B.bmp"
    topPercentToSaturate = 1
    thresholdingStrategy = 'otsu'
    objectMinSize = 256
    # Reading Image
    preImage = Image.open(dataPath_A)
    postImage = Image.open(dataPath_B)
    preChangeImage = np.array(preImage)
    postChangeImage = np.array(postImage)
    print(preChangeImage.shape)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    ##Reassigning pre-change and post-change image to normalized values
    data1 = np.copy(preChangeImage)
    data2 = np.copy(postChangeImage)
    # Checking image dimension
    imageSize = data1.shape
    imageSizeCol = imageSize[0]
    imageSizeRow = imageSize[1]
    imageNumberOfChannel = imageSize[2]

    # patch
    eachPatch = 256
    numPatchRow = int(imageSizeRow / eachPatch)
    numPatchCol = int(imageSizeCol / eachPatch)
    additionalPatchPixel = 256
    data_pre = data1.transpose((2, 0, 1))
    data_pos = data2.transpose((2, 0, 1))
    if ((imageSizeRow > eachPatch * numPatchRow) | (imageSizeCol > eachPatch * numPatchCol)):
        data1 = np.pad(data1, ((0, eachPatch - (imageSizeCol - eachPatch * numPatchCol)),
                               (0, eachPatch - (imageSizeRow - eachPatch * numPatchRow)), (0, 0)), 'symmetric')
        data2 = np.pad(data2, ((0, eachPatch - (imageSizeCol - eachPatch * numPatchCol)),
                               (0, eachPatch - (imageSizeRow - eachPatch * numPatchRow)), (0, 0)), 'symmetric')

    data1 = data1.transpose((2, 0, 1))
    data2 = data2.transpose((2, 0, 1))
    print(data1.shape)
    # data1
    dataheight = data1.shape[1]
    datawidth = data1.shape[2]
    numCol = int(dataheight / eachPatch)
    numRow = int(datawidth / eachPatch)
    image1 = []
    image2 = []
    for i in range(0, numCol * numRow):
        image1.append([])
        image2.append([])
    count = 0
    data_transform = transform.Compose([transform.RandomCrop((256, 256)), transform.ToTensor(),
                                        transform.Normalize(mean, std)])
    data_transform_test = transform.Compose([transform.ToTensor(),
                                        transform.Normalize(mean, std)])
    for i in range(numCol):
        for j in range(numRow):
            mm = ((i + 1) * eachPatch)
            nn = ((j + 1) * eachPatch)
            img1 = data1[:, i * eachPatch:mm, j * eachPatch:nn]
            img1 = img1.transpose((1,2,0))
            img1 = Image.fromarray(img1)
            img1_tensor = data_transform(img1)
            img1_tensor = torch.unsqueeze(img1_tensor,0)
            image1[count] = img1_tensor
            img2 = data2[:, i * eachPatch:mm, j * eachPatch:nn]
            img2 = img2.transpose((1, 2, 0))
            img2 = Image.fromarray(img2)
            img2_tensor = data_transform(img2)
            img2_tensor = torch.unsqueeze(img2_tensor, 0)
            image2[count] = img2_tensor
            count += 1
    # crop for test
    def TifCroppingArray(img_pre, img_pos, Length, SideLength):
        ArrayReturn_pre = []
        ArrayReturn_pos = []
        ColumnNum = int((img_pre.shape[0] - SideLength * 2) / (Length - SideLength * 2))
        RowNum = int((img_pre.shape[1] - SideLength * 2) / (Length - SideLength * 2))
        for i in range(ColumnNum):
            Array_pre = []
            Array_pos = []
            for j in range(RowNum):
                cropped_pre = img_pre[i * (Length - SideLength * 2): i * (Length - SideLength * 2) + Length,
                              j * (Length - SideLength * 2): j * (Length - SideLength * 2) + Length]
                cropped_pos = img_pos[i * (Length - SideLength * 2): i * (Length - SideLength * 2) + Length,
                              j * (Length - SideLength * 2): j * (Length - SideLength * 2) + Length]
                Array_pre.append(cropped_pre)
                Array_pos.append(cropped_pos)
            ArrayReturn_pre.append(Array_pre)
            ArrayReturn_pos.append(Array_pos)
        for i in range(ColumnNum):
            cropped_pre = img_pre[i * (Length - SideLength * 2): i * (Length - SideLength * 2) + Length,
                          (img_pre.shape[1] - Length): img_pre.shape[1]]
            cropped_pos = img_pos[i * (Length - SideLength * 2): i * (Length - SideLength * 2) + Length,
                          (img_pos.shape[1] - Length): img_pos.shape[1]]
            ArrayReturn_pre[i].append(cropped_pre)
            ArrayReturn_pos[i].append(cropped_pos)
        Array_pre = []
        Array_pos = []
        for j in range(RowNum):
            cropped_pre = img_pre[(img_pre.shape[0] - Length): img_pre.shape[0],
                          j * (Length - SideLength * 2): j * (Length - SideLength * 2) + Length]
            cropped_pos = img_pos[(img_pos.shape[0] - Length): img_pos.shape[0],
                          j * (Length - SideLength * 2): j * (Length - SideLength * 2) + Length]
            Array_pre.append(cropped_pre)
            Array_pos.append(cropped_pos)
        cropped_pre = img_pre[(img_pre.shape[0] - Length): img_pre.shape[0],
                      (img_pre.shape[1] - Length): img_pre.shape[1]]
        cropped_pos = img_pos[(img_pos.shape[0] - Length): img_pos.shape[0],
                      (img_pos.shape[1] - Length): img_pos.shape[1]]
        Array_pre.append(cropped_pre)
        Array_pos.append(cropped_pos)
        ArrayReturn_pre.append(Array_pre)
        ArrayReturn_pos.append(Array_pos)
        ColumnOver = (img_pre.shape[0] - SideLength * 2) % (Length - SideLength * 2) + SideLength
        RowOver = (img_pre.shape[1] - SideLength * 2) % (Length - SideLength * 2) + SideLength
        return ArrayReturn_pre, ArrayReturn_pos, RowOver, ColumnOver
    def Result(shape, TifArray, npyfile, Length, RepetitiveLength, RowOver, ColumnOver):
        result = np.zeros(shape, np.uint8)
        j = 0
        for i, item in enumerate(npyfile):
            img = item.astype(np.uint8)
            if (i % len(TifArray[0]) == 0):
                if (j == 0):
                    result[0: Length - RepetitiveLength, 0: Length - RepetitiveLength] = img[
                                                                                         0: Length - RepetitiveLength,
                                                                                         0: Length - RepetitiveLength]
                elif (j == len(TifArray) - 1):
                    result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0: Length - RepetitiveLength] = img[
                                                                                                               Length - ColumnOver - RepetitiveLength: Length,
                                                                                                               0: Length - RepetitiveLength]
                else:
                    result[j * (Length - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                            Length - 2 * RepetitiveLength) + RepetitiveLength,
                    0:Length - RepetitiveLength] = img[RepetitiveLength: Length - RepetitiveLength,
                                                   0: Length - RepetitiveLength]
            elif (i % len(TifArray[0]) == len(TifArray[0]) - 1):
                if (j == 0):
                    result[0: Length - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[
                                                                                         0: Length - RepetitiveLength,
                                                                                         Length - RowOver: Length]
                elif (j == len(TifArray) - 1):
                    result[shape[0] - ColumnOver: shape[0], shape[1] - RowOver: shape[1]] = img[
                                                                                            Length - ColumnOver: Length,
                                                                                            Length - RowOver: Length]
                else:
                    result[j * (Length - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                            Length - 2 * RepetitiveLength) + RepetitiveLength,
                    shape[1] - RowOver: shape[1]] = img[RepetitiveLength: Length - RepetitiveLength,
                                                    Length - RowOver: Length]
                j = j + 1
            else:
                if (j == 0):
                    result[0: Length - RepetitiveLength,
                    (i - j * len(TifArray[0])) * (Length - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                        TifArray[0]) + 1) * (Length - 2 * RepetitiveLength) + RepetitiveLength
                    ] = img[0: Length - RepetitiveLength, RepetitiveLength: Length - RepetitiveLength]
                if (j == len(TifArray) - 1):
                    result[shape[0] - ColumnOver: shape[0],
                    (i - j * len(TifArray[0])) * (Length - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                        TifArray[0]) + 1) * (Length - 2 * RepetitiveLength) + RepetitiveLength
                    ] = img[Length - ColumnOver: Length, RepetitiveLength: Length - RepetitiveLength]
                else:
                    result[j * (Length - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                            Length - 2 * RepetitiveLength) + RepetitiveLength,
                    (i - j * len(TifArray[0])) * (Length - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                        TifArray[0]) + 1) * (Length - 2 * RepetitiveLength) + RepetitiveLength,
                    ] = img[RepetitiveLength: Length - RepetitiveLength, RepetitiveLength: Length - RepetitiveLength]
        return result
    def filtering(d, pre, pos):
        # print("Filtering!")
        d = d[..., np.newaxis]
        d = np.concatenate((d, 1.0 - d), axis=2)
        W = np.size(d, 0)
        H = np.size(d, 1)
        stack = np.concatenate((pre, pos), axis=2)
        CD = dcrf.DenseCRF2D(W, H, 2)
        d[d == 0] = 10e-20
        U = -(np.log(d))
        U = U.transpose(2, 0, 1).reshape((2, -1))
        U = U.copy(order="C")
        CD.setUnaryEnergy(U.astype(np.float32))
        pairwise_energy_gaussian = create_pairwise_gaussian((10, 10), (W, H))
        CD.addPairwiseEnergy(pairwise_energy_gaussian, compat=1)
        pairwise_energy_bilateral = create_pairwise_bilateral(
            sdims=(10, 10), schan=(0.1,), img=stack, chdim=2
        )
        CD.addPairwiseEnergy(pairwise_energy_bilateral, compat=1)
        Q = CD.inference(3)
        heatmap = np.array(Q)
        heatmap = np.reshape(heatmap[0, ...], (W, H))
        return heatmap
    # train network
    print(len(image1), len(image2))


    # prepare model
    trainer = Grid(opts, gpu_num).cuda()
    resume = 'Path/models/gen_00794001.pt'
    trainer.resume_eval(resume)

    torch.backends.cudnn.benchmark = True

    # start training
    total_images = len(image1)
    print('-' * 8 + 'Start training' + '-' * 8)
    initial_step = trainer.resume(checkpoint_directory, opts) if opts.resume else 0
    total_step = total_images // opts.batch_size
    step = initial_step
    for iteration in range(opts.epoch):
        for batch_num, image in enumerate((zip(image1, image2))):
            image_pre = image[0]
            image_pos = image[1]
            image_pre = image_pre.float()
            image_pos = image_pos.float()
            t0 = datetime.datetime.now()
            step +=1
            content_cuda = put_tensor_cuda(image_pre)
            style_cuda = put_tensor_cuda(image_pos)
            trainer.update_learning_rate()

            # training update
            trainer.update(content_cuda, style_cuda, opts)
            batch_output = trainer.get_output()
            batch_content_style = trainer.get_content_style()
            display = torch.cat([batch_content_style[:1], batch_output[:1]], 3)
            if step % 500 == 0:
                write_loss(step, trainer, train_writer)
            if step % 500 == 0:
                write_images('content_style_output', display, train_writer, step)
            if step % 500 == 0:
                result = torchvision.utils.make_grid(denormalize_vgg_adain(display).cpu())
                torchvision.utils.save_image(result, os.path.join(image_directory, 'test_%08d.jpg' % (total_step + 1)))
            if step % opts.save_freq == 0:
                trainer.save(checkpoint_directory, step)
            t1 = datetime.datetime.now()
            time = t1 - t0
            if step % 50 == 0:
                print("Epoch: %08d/%08d, iteration: %08d/%08d time: %.8s gloss = %.8s  conloss = %.8f" % (
                    iteration + 1, opts.epoch, step, total_step,
                    time.seconds + 1e-6 * time.microseconds, trainer.gener_loss.item(), trainer.per_loss.item(),
                    ))
            if step % 500 == 0:
                if not os.path.exists(
                        "Path/Output/visualization"):
                    os.makedirs(
                        "Path/Output/visualization")
                count = 0
                imge = []
                imge2 = []
                y_hat = []
                for i in range(0, numCol * numRow):
                    imge.append([])
                    imge2.append([])
                    y_hat.append([])
                for i in range(numCol):
                    for j in range(numRow):
                        mm = ((i + 1) * eachPatch)
                        nn = ((j + 1) * eachPatch)
                        if ((i + 1) * eachPatch) > imageSizeCol:
                            mm = imageSizeCol
                        if (j + 1) * eachPatch > imageSizeRow:
                            nn = imageSizeRow
                        imge[count] = data_pre[:, i * eachPatch:mm, j * eachPatch:nn]
                        imge2[count] = data_pos[:, i * eachPatch:mm, j * eachPatch:nn]
                        img1 = imge[count]
                        img1 = img1.transpose((1, 2, 0))
                        img1 = Image.fromarray(img1)
                        img1_tensor = data_transform_test(img1)
                        img1_tensor = torch.unsqueeze(img1_tensor, 0)

                        img2 = imge2[count]
                        img2 = img2.transpose((1, 2, 0))
                        img2 = Image.fromarray(img2)
                        img2_tensor = data_transform_test(img2)
                        img2_tensor = torch.unsqueeze(img2_tensor, 0)

                        content_pre_cuda = put_tensor_cuda(img1_tensor.float())
                        style_pos_cuda = put_tensor_cuda(img2_tensor.float())

                        with torch.no_grad():
                            image_hat = trainer.sample(content_pre_cuda, style_pos_cuda)
                        image_hat = denormalize_vgg(image_hat).clamp_(0., 255.)
                        image_hat = image_hat.data.cpu().numpy()
                        image_hat = image_hat.squeeze()
                        if count % numRow == 0:
                            y_hat[i] = image_hat
                        else:
                            y_hat[i] = np.concatenate((y_hat[i], image_hat), axis=2)
                        count += 1
                y_hatm = y_hat[0]
                for i in range(0, numCol - 1):
                    y_hatm = np.concatenate((y_hatm, y_hat[i + 1]), axis=1)
                print(y_hatm.shape)
                y_hatm = y_hatm.transpose((1, 2, 0))
                y_hat_image = np.array(y_hatm).astype(np.uint8)
                y_hat_image = Image.fromarray(y_hat_image)
                y_hat_image.save("Path/Output/visualization/%d.png" % step)
                print("images saved")
            if step % 500 == 0:
                count = 0
                image_pr = []
                image_po = []
                binarymap = []
                Unbinarymap = []
                Toclass = []
                # data
                Length = 256
                area_perc = 0.5
                RepetitiveLength = int((1 - math.sqrt(area_perc)) * Length / 2)
                pre_data = data_pre.transpose(1, 2, 0)
                pos_data = data_pos.transpose(1, 2, 0)
                ArrayReturn_pre, ArrayReturn_pos, RowOver, ColumnOver = TifCroppingArray(pre_data, pos_data, Length, RepetitiveLength)

                for i in range(len(ArrayReturn_pre)):
                    for j in range(len(ArrayReturn_pre[0])):
                        image_pre = ArrayReturn_pre[i][j]
                        image_pre = Image.fromarray(image_pre)
                        image_pre = data_transform_test(image_pre)
                        image_pre_m = np.array(image_pre)
                        image_pre_mm = image_pre_m.transpose(1,2,0)
                        # print('image_pre_mm', image_pre_mm.shape)
                        image_pre = torch.unsqueeze(image_pre, 0)
                        image_pos = ArrayReturn_pos[i][j]

                        image_pos = Image.fromarray(image_pos)
                        image_pos = data_transform_test(image_pos)
                        image_pos_m = np.array(image_pos)
                        image_pos_mm = image_pos_m.transpose(1,2,0)
                        image_pos = torch.unsqueeze(image_pos, 0)
                        content_pre_cuda = put_tensor_cuda(image_pre.float())
                        style_pos_cuda = put_tensor_cuda(image_pos.float())

                        # model
                        with torch.no_grad():
                            image_hat_pre = trainer.sample(content_pre_cuda, style_pos_cuda)
                        image_pre_features = trainer.vgg(image_hat_pre)
                        image_pos_features = trainer.vgg(style_pos_cuda)
                        image_pre_features[1] = F.interpolate(image_pre_features[1],
                                                              size=image_pre_features[0].size()[2:],
                                                              mode='bilinear', align_corners=True)
                        image_pre_features[2] = F.interpolate(image_pre_features[2],
                                                              size=image_pre_features[0].size()[2:],
                                                              mode='bilinear', align_corners=True)
                        image_pre_features[3] = F.interpolate(image_pre_features[3],
                                                              size=image_pre_features[0].size()[2:],
                                                              mode='bilinear', align_corners=True)
                        image_pos_features[1] = F.interpolate(image_pos_features[1],
                                                              size=image_pos_features[0].size()[2:],
                                                              mode='bilinear', align_corners=True)
                        image_pos_features[2] = F.interpolate(image_pos_features[2],
                                                              size=image_pos_features[0].size()[2:],
                                                              mode='bilinear', align_corners=True)
                        image_pos_features[3] = F.interpolate(image_pos_features[3],
                                                              size=image_pos_features[0].size()[2:],
                                                              mode='bilinear', align_corners=True)
                        # take absolute value for binary CD
                        for m in range(4):
                            image_pre_features[m] = image_pre_features[m].data.cpu().numpy()
                            image_pos_features[m] = image_pos_features[m].data.cpu().numpy()
                            ##Normalize the features (separate for both images)
                            image_pre_features[m] = np.squeeze(image_pre_features[m])
                            image_pos_features[m] = np.squeeze(image_pos_features[m])
                            if m == 0:
                                pre_features = image_pre_features[m]
                                pos_features = image_pos_features[m]
                            else:
                                pre_features = np.concatenate((pre_features, image_pre_features[m]), axis=0)
                                pos_features = np.concatenate((pos_features, image_pos_features[m]), axis=0)
                        axis = pre_features.shape[0]
                        map = np.absolute(saturateImage().saturateSomePercentileMultispectral(pre_features, 5, axis) - \
                                           saturateImage().saturateSomePercentileMultispectral(pos_features, 5, axis))
                        changemap = np.linalg.norm(map, axis=(0))
                        detectedChangeMapNormalized = (changemap - np.amin(changemap)) / (
                                    np.amax(changemap) - np.amin(changemap) + np.exp(-10))
                        #cluster
                        detectedChangeMapNormalizedX = detectedChangeMapNormalized.reshape(-1, 1)
                        fcm = FCM(n_clusters=3)
                        fcm.fit(detectedChangeMapNormalizedX)
                        # print(detectedChangeMapNormalizedX.shape)
                        fcm_labels = fcm.predict(detectedChangeMapNormalizedX)
                        fcm_label = fcm_labels.reshape([detectedChangeMapNormalized.shape[0], detectedChangeMapNormalized.shape[1]])
                        fcm_label = fcm_label.squeeze()
                        fcm_centers = fcm.centers
                        Changelabel = np.argmax(fcm_centers)
                        Unchangelabel = np.argmin(fcm_centers)
                        Changemap = np.where(fcm_label==Changelabel, 1, 0)
                        Uchangemap = np.where(fcm_label==Unchangelabel, 0, 1)
                        #change map2
                        cdMap = np.zeros(detectedChangeMapNormalized.shape, dtype=bool)
                        Ones = np.ones(detectedChangeMapNormalized.shape)
                        otsuScalingFactor = 1.25
                        for sigma in range(101, 202, 50):
                            adaptiveThreshold = 2 * filters.gaussian(detectedChangeMapNormalized, sigma)
                            cdMapTemp = (detectedChangeMapNormalized > adaptiveThreshold)
                            cdMapTemp = morphology.remove_small_objects(cdMapTemp, min_size=objectMinSize)
                            cdMap = cdMap | cdMapTemp
                        cdMap = morphology.binary_closing(cdMap, morphology.disk(3))
                        cdMap = cdMap.squeeze()
                        # print('Changemap', Changemap.shape)
                        # print('cdMap', cdMap.shape)
                        Change = np.divide((Changemap + cdMap),2).astype(int)
                        Uchange = np.where(Uchangemap + cdMap==0, 1, 0)
                        Toclassified = Ones - Change - Uchange

                        binarymap.append(Change)
                        Unbinarymap.append(Uchange)
                        Toclass.append(Toclassified)
                shape = (pre_data.shape[0], pre_data.shape[1])
                binarymapm = Result(shape, ArrayReturn_pre, binarymap, Length, RepetitiveLength, RowOver, ColumnOver)
                Unbinarymap = Result(shape, ArrayReturn_pre, Unbinarymap, Length, RepetitiveLength, RowOver, ColumnOver)
                Tobecla = Result(shape, ArrayReturn_pre, Toclass, Length, RepetitiveLength, RowOver, ColumnOver)
                #train VGG
                #data
                imge_pre_change = []
                image_pos_change = []
                image_change = []
                imge_pre_unchange = []
                imge_pos_unchange = []
                image_unchange = []
                label_change = []
                label_unchange = []
                change_num = np.count_nonzero(binarymapm == 1)
                unchange_num = np.count_nonzero(Unbinarymap == 1)
                for m in range(0, change_num):
                    imge_pre_change.append([])
                    image_pos_change.append([])
                    image_change.append([])
                    label_change.append([])
                for n in range(0, unchange_num):
                    imge_pre_unchange.append([])
                    imge_pos_unchange.append([])
                    image_unchange.append([])
                    label_unchange.append([])
                countc = 0
                countu = 0
                Patch = 17
                pad_width = int((Patch + 1) / 2)
                for i in range(binarymapm.shape[0]):
                    for j in range(binarymapm.shape[1]):
                        if binarymapm[i, j] == 1:
                            imge_pre_change[countc] = data_pre[:, i, j]
                            image_pos_change[countc] = data_pos[:, i, j]
                            img1 = imge_pre_change[countc]
                            image_change[countc] = np.concatenate((imge_pre_change[countc], image_pos_change[countc]), 0)
                            label_change[countc] = np.ones(1)
                            countc = countc + 1
                        if Unbinarymap[i,j] == 1:
                            imge_pre_unchange[countu] = data_pre[:, i, j]
                            imge_pos_unchange[countu] = data_pos[:, i, j]
                            image_unchange[countu] = np.concatenate((imge_pre_unchange[countu], imge_pos_unchange[countu]), 0)
                            label_unchange[countu] = np.zeros(1)
                            countu = countu + 1
                len_c = len(image_change)
                images = image_change + image_unchange
                labels = label_change + label_unchange
                print('len of change images:', len(image_change), 'len of unchange images:', len(image_unchange[:len_c]))
                print('len of images:', len(images), 'len of labels:', len(labels))
                data = list(zip(images, labels))
                shuffle(data)
                images, labels = zip(*data)
                # tensor
                images = np.stack(images, axis=0)
                labels = np.stack(labels, axis=0)
                print(images.shape)
                # images = np.squeeze(images)
                labels = np.squeeze(labels)
                rf = RandomForestClassifier()
                rf.fit(images, labels)
                #inference
                for i in range(binarymapm.shape[0]):
                    for j in range(binarymapm.shape[1]):
                        if Tobecla[i, j] == 1:
                            img1 = data_pre[:, i, j]
                            img2 = data_pos[:, i, j]
                            image = np.concatenate((img1, img2), 0)
                            image = image.reshape(1, -1)
                            predict = rf.predict(image)
                            # print('predict:', predict)
                            if predict == 1:
                                binarymapm[i,j] = 1
                binarymapmm = np.zeros(binarymapm.shape, dtype=bool)
                for i in range(binarymapm.shape[0]):
                    for j in range(binarymapm.shape[1]):
                        if binarymapm[i,j] == 1:
                            binarymapmm[i, j] = True
                        else:
                            binarymapmm[i, j] = False
                binarymapmm = morphology.remove_small_objects(binarymapmm, min_size=objectMinSize)
                binarymapmm = morphology.binary_closing(binarymapmm, morphology.disk(3))
                binarymap = binarymapmm.squeeze()
                print(binarymap.shape)
                binarymapm = Image.fromarray(binarymap)
                Unbinarymap = Image.fromarray(Unbinarymap)
                Tobecla = Image.fromarray(Tobecla)
                binarymapm.save(
                    "Path/1.png")
                Unbinarymap.save(
                    "Path/2.png")
                Tobecla.save(
                    "Path/3.png")
                print("images saved")
                # accuracy obtained
                gt = Image.open("Path/Seasonal_Changes/label.png")
                result = np.array(binarymapm, dtype=np.uint8)
                gt = np.array(gt, dtype=np.uint8)
                gt[gt != 1] = 0
                all_size = gt.size
                tp = np.count_nonzero((gt == result) & (gt > 0))
                tn = np.count_nonzero((gt == result) & (gt == 0))
                fp = np.count_nonzero(gt < result)
                fn = np.count_nonzero(gt > result)
                a1 = changes = tp + fn
                a2 = unchanges = fp + tn
                b1 = tp + fp
                b2 = fn + tn
                # change
                Recall_change = tp * 1.0 / changes
                Precision_change = tp * 1.0 / b1
                misdetection_change = fn * 1.0 / changes
                falsealarms_change = fp * 1.0 / unchanges
                # unchange
                Recall_unchange = tn * 1.0 / unchanges
                Precision_unchange = tn * 1.0 / b2
                # average
                accuray = (tp + tn) * 1.0 / all_size
                F1_Score_b = mt.f1_score(gt.flatten(), result.flatten())
                KC_b = mt.cohen_kappa_score(gt.flatten(), result.flatten())
                overallerror = (fp + fn) * 1.0 / all_size
                "--------------Accuracy---------------"
                print("     Recall_change: {0:.2f} %".format(Recall_change * 100))
                print("     Precision_change: {0:.2f} %".format(Precision_change * 100))
                print("     misdetection_change: {0:.2f} %".format(misdetection_change * 100))
                print("     falsealarms_change: {0:.2f} %".format(falsealarms_change * 100))
                print("     Recall_unchange: {0:.2f} %".format(Recall_unchange * 100))
                print("     Precision_unchange: {0:.2f} %".format(Precision_unchange * 100))
                print("    overall accuray(OA): {0:.2f} %".format(accuray * 100))
                print("    F1_Score_b(F1): {0:.2f} %".format(F1_Score_b * 100))
                print("    KC_b(Kappa): {0:.2f} %".format(KC_b * 100))
                print("    overall error(OE): {0:.2f} %".format(overallerror * 100))
                "-------------------------------------"
    trainer.save(checkpoint_directory, step)
    print("Training is finished.")




