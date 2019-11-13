import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import animation


################################
# 修改这个alpha玩！其他代码不需要动！
#　这个示意的例子跟真实ML不一样，alpha通常在0.001到0.1之间
alpha = 0.9
################################


x = np.arange(-5,5,0.1)
y = x**2

fig, ax = plt.subplots()
ax.grid()
ax.plot(x,y)

# y = x ^ 2
pointX,pointY = 2,4
point = plt.scatter(pointX,pointY)

def data_gen():
	x = pointX
	y = pointY

	dx,dy = 0,0

	t = 0
	while t < 150 and y < 25:
		if t % 10 == 0:
			x -= dx
			y -= dy

			diff = 2 * x
			dx = diff * alpha
			dy = y - (x - dx)**2

		rate = (t%10+1)/10
		dxt = dx * rate
		dyt = dy * rate

		yield x,y,-dxt,-dyt
		t += 1

def update(data):
	x,y,dxt,dyt = data
	point.set_offsets((x,y))
	arr = ax.arrow(*data,head_width=0.5, head_length=0.15,fc='r',ec='r')
	print(x,y,dxt,dyt)
	return point,arr,
ani = animation.FuncAnimation(fig,update,data_gen,interval=80,blit=True)
plt.show()






# 反注释以下代码，运行可以得到MP4输出
# 
# from moviepy.editor import VideoClip
# from moviepy.video.io.bindings import mplfig_to_npimage
#
# dg = data_gen()
# def makeFrame(t):
# 	data = next(dg)
# 	update(data)
# 	return mplfig_to_npimage(fig)

# ani = VideoClip(makeFrame,duration=150)
# ani.write_videofile("lrBig.mp4", fps=20)