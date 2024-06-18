def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'posenet':
        from .posenet_model import PoseNetModel
        model = PoseNetModel()
    elif opt.model == 'poselstm':
        from .poselstm_model import PoseLSTModel
        model = PoseLSTModel()
    elif opt.model == 'resnet50':
        from .posenet_model import ResNet50PoseModel
        model = ResNet50PoseModel()
    elif opt.model == 'resnet101':
        from .posenet_model import ResNet101PoseModel
        model = ResNet101PoseModel()
    elif opt.model == 'vgg16':
        from .posenet_model import VGG16PoseModel
        model = VGG16PoseModel()
    elif opt.model == 'vgg19':
        from .posenet_model import VGG19PoseModel
        model = VGG19PoseModel()
    elif opt.model == 'efficientnetB0':
        from .posenet_model import EfficientNetB0PoseModel
        model = EfficientNetB0PoseModel()
    elif opt.model == 'mobilenetV2':
        from .posenet_model import MobileNetV2PoseModel
        model = MobileNetV2PoseModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model


