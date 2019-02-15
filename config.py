trainpath = r"D:\Users\yl_gong\Desktop\dl\voc\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages"
styleimg = r'D:\Users\yl_gong\Desktop\rain_princess.jpg'
modelpath = r"D:\Users\yl_gong\Desktop\ckp\ckp"
testimg = r"D:\Users\yl_gong\Desktop\chicago.jpg"
testpath = r"D:\Users\yl_gong\Desktop\tp"
savepath = r"D:\Users\yl_gong\Desktop\sp"
epoch = 2
batchsize = 4

styletensors = [
           "MobilenetV2/expanded_conv_6/depthwise/Relu6:0",
           "MobilenetV2/expanded_conv_12/depthwise/Relu6:0",
           "MobilenetV2/expanded_conv_9/depthwise/Relu6:0"
           ]
content = r'MobilenetV2/expanded_conv_10/expand/Relu6:0'