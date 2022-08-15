import time
import visdom
import numpy as np


class Visualizer(object):
    """
    封装visdom的基本操作
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        self.index = {}
        self.log_text = ''

    def reinit(self, env="default", **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name, value) i.e. ('loss', 0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        """
        一次img_many
        @params d: dict (name, value) i.e. ('image', img_tensor)
        """
        for k, v in d.items():
            self.img(k, v)

    def plot(self, win, name, y, **kwargs):
        """
        self.plot('loss', 1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(
            Y=np.array([y]),
            X=np.array([x]),
            win=win,
            name=name,
            opts=dict(
                title=win,
                showlegend=False,  # 显示网格
                xlabel='x1',  # x轴标签
                ylabel='y1',  # y轴标签
                fillarea=False,  # 曲线下阴影覆盖
                width=2400,  # 画布宽
                height=350,  # 画布高
            ),
            update=None if x == 0 else 'append',
            **kwargs
        )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img, torch.Tensor(64,64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)

        !!! don't ~~self.img('input_imgs', t.Tensor(100, 64, 64), nrows=10)~~ !!!
        """
        self.vis.images(img_.cpu().numpy(), win=name,
                        opts=dict(title=name), **kwargs)

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1, 'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        """
        自定义的plot,image,log,plot_many等除外
        self.function 等价于self.vis.function
        """
        return getattr(self.vis, name)
