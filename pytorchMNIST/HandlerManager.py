from MyHandler import MyHandler

_service = MyHandler()


def handle(input, context):
    if not _service.initialized:
        _service.initialize(context)

    if input is None:
        return None

    input = _service.preprocess(input)
    data = _service.inference(input)
    output = _service.postprocess(data)
    return output