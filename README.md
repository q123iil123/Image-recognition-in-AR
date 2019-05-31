# Image-recognition-in-AR
这个库用来记录我探索AR中的图像识别的过程

AR流程的第一步是图像识别，首先你要通过特征提取和匹配的算法识别出你想要增强的图片，然后再对匹配到的关键点求单应性映射，之后在三维空间中渲染你要增强的内容

图像识别是一个难点，难点在于任意给你一张target图片，你都要对此图片打分，如果得分高的话，意味着你要能在任意复杂场景中识别出这张图片

这无疑是困难的，虽然有打分这个环节，但你要识别的图片可以是任意的。用户向你的服务器上传了一张作为target的图片之后，你的AR软件就要能够立刻能在空间中识别出这张图片，这之中存在着很多工程的问题

可以明确的是单纯的特征提取和匹配是绝对无法做到这样的图像识别的，这是因为会有大量的误匹配，甚至可能比正确的匹配还要多。

现在流行的基于深度学习的方法也被我否决了，这是因为深度学习虽然有着极高的识别率，但往往只能识别其训练样本内的数据，假如我上传一张你训练样本中没有的场景的图片的话，深度学习是没办法解决这个问题的。


当前的思路是使用AKAZE，结合kmeans树求近似最临匹配，然后再对匹配到的关键点求单应性映射，最后根据inlier点集打分的方式来最终判定识别到的目标。

feature_matching.cpp：
    
    该程序实现了任意两张图像之间的特征匹配的功能，使用的是AKAZE特征，程序框架参考了高博的视觉SLAM十四讲中的特征提取和特征匹配的部分，加入了RANSAC来筛选误匹配点
    该程序只实现了匹配的功能，无法进行图像识别
