# 车道线检测与车辆检测

本项目基于 [keras-yolo3](https://github.com/qqwweee/keras-yolo3)和[CarND-Advanced-Lane-Lines](https://github.com/udacity/CarND-Advanced-Lane-Lines)

## 快速开始

1. 下载yolo的keras模型放在yolo/model_data文件夹中。下载：[百度网盘](https://pan.baidu.com/s/1GQPt0xM2tkygJ7qp6M2cpw)，密码：2hhv；OneDrive：[yolo.h5](https://csueducn-my.sharepoint.com/personal/yinaoxiong_csu_edu_cn/_layouts/15/download.aspx?e=FHhthG&share=EUt0K3xZNQxGhTO4lkv0vPABd3CIXcE_reWc8C6sqTl5xg&cid=5663eb40-2260-41f2-a1c9-f318e5be185d) , [yolo-tiny.h5](https://csueducn-my.sharepoint.com/personal/yinaoxiong_csu_edu_cn/_layouts/15/download.aspx?e=XtyO7d&share=EWePbN2yQ1VIi7nw5vL_4JkB07JZmm0ZynbTo1zoQ8jJTg&cid=10e4d16e-9cc3-42b9-bd9c-7126372bed9e)

2. 安装依赖

   ```
   conda install tensorflow-gpu
   pip install -r requirements.txt
   ```

   

3. 运行main.py进行视频检测，例子：
   `python main.py -i test_videos/test_video.mp4 -o results/test_video_result.mp4`

## 使用说明

使用--help查看main.py的使用说明

```
usage: main.py [-h] [-i [INPUT]] [-o [OUTPUT]] [--show] [--fourcc FOURCC]

optional arguments:
  -h, --help            show this help message and exit
  -i [INPUT], --input [INPUT]
                        Video input path
  -o [OUTPUT], --output [OUTPUT]
                        [Optional] Video output path
  --show                Demonstrate the process of processing video
  --fourcc FOURCC       an identifier for a video codec, compression format,color or pixel format used in media files
```

