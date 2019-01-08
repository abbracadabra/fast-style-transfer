trainpath = r"D:\Users\yl_gong\Desktop\dl\voc\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages"
styleimg = r'D:\Users\yl_gong\Desktop\rain_princess.jpg'
modelpath = r"D:\Users\yl_gong\Desktop\ckp"
testimg = r"D:\Users\yl_gong\Desktop\tp\000037.jpg"
testpath = r"D:\Users\yl_gong\Desktop\tp"
savepath = r"D:\Users\yl_gong\Desktop\sp"
epoch = 1
batchsize = 1

styletensors = [
           "MobilenetV2/expanded_conv_3/expand/Relu6:0",
           "MobilenetV2/expanded_conv_6/expand/Relu6:0",
           "MobilenetV2/expanded_conv_9/expand/Relu6:0",
           "MobilenetV2/expanded_conv_12/expand/Relu6:0"
           ]
content = r'MobilenetV2/expanded_conv_10/expand/Relu6:0'