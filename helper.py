from distutils.log import error
from cv2 import imread
from pandas import DataFrame
from pytesseract import image_to_string
from torch import hub

def modelLoad(path):
    ''' Load Yolo Custom Model '''
    model = hub.load('ultralytics/yolov5', 'custom', path=path)
    return model

def imageImport(path:str)-> list:
    ''' Return OpenCv image'''
    img = imread(path)
    return img

def areaDetection(img, model):
    ''' Return yolo Text area detection in pandas '''
    results = model(img)
    data = results.pandas().xyxy[0]
    df = DataFrame(data)
    dfFinal = df.groupby("class", as_index=False).max()
    return dfFinal

def cropImages(df, image)-> list:
    ''' Take df and return multiple crop images array'''
    cropImages = list()
    for i in df.index:
        data = df[df["class"] == df["class"][i]]
        tempCrop = image[int(data["ymin"][i]):int(data["ymax"][i]), int(data["xmin"][i]):int(data["ymax"][i])]
        cropImages.append(tempCrop)
    return cropImages

def textDetection(images:list) -> list:
    text = ''
    for i in images:
        text += '' + image_to_string(i)
    return text


if __name__ == "__main__":
    imageImport()
    areaDetection()
    cropImages()
    textDetection()
    modelLoad()

# def cropImages(df, image)-> list:
#     ''' Take df and return multiple crop images array'''
#     cropImages = list()
#     for i in range(0,3):
#         data = df[df["class"] == i]
#         tempCrop = image[int(data["ymin"][i]):int(data["ymax"][i]), int(data["xmin"][i]):int(data["ymax"][i])]
#         cropImages.append(tempCrop)
#     return cropImages


# def textDetection(images:list) -> list:
#     text = ''
#     for i in images:
#         text += '' + image_to_string(i)
#     return text
