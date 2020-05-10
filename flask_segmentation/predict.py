
from flask import Flask, render_template, request, redirect, url_for, abort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from datetime import datetime
import numpy as np


from utils.dataloader import make_datapath_list, DataTransform
from utils.pspnet import PSPNet



device = torch.device("cpu")

model = PSPNet(n_classes=21).to(device)
# 学習モデルをロードする
model.load_state_dict(
    torch.load("pspnet50_30.pth", map_location=lambda storage, loc: storage)
)
model = model.eval()



app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        # アプロードされたファイルをいったん保存する
        f = request.files["file"]
        filepath = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        f.save(filepath)

        # 画像ファイルを読み込む&前処理 
        image = Image.open(filepath)
        img_width, img_height = image.size

        color_mean = (0.485, 0.456, 0.406)
        color_std = (0.229, 0.224, 0.225)
        transform = DataTransform(
            input_size=475, color_mean=color_mean, color_std=color_std)
        
        anno_class_img = Image.open('./static/palette.png')   # [高さ][幅]
        p_palette = anno_class_img.getpalette()
        phase = "val"
        image, anno_class_img = transform(phase, image, anno_class_img)

        # 予測を実施
        x = image.unsqueeze(0)
        output = model(x)
        y = output[0]
        y = y[0].detach().numpy()  # y：torch.Size([1, 21, 475, 475])
        y = np.argmax(y, axis=0)
        anno_class_img = Image.fromarray(np.uint8(y), mode="P")
        anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
        anno_class_img.putpalette(p_palette)
        filepath = './static/' + filepath[9:]
        anno_class_img.save(filepath)

        return render_template("index.html", 
            filepath=filepath, result=1)


if __name__ == "__main__":
    app.run(debug=True)