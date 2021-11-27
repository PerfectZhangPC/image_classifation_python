import tf.lite as lite

saved_model_dir = "C:/Users/47342/Desktop/image_classfication/model_weight/mobilenetV2_fruit_flyNet.pb"
converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()