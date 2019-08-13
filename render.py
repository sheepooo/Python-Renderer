from tkinter import  *
from PIL import Image,ImageTk
import math
import copy
import threading
import random
import time

WIDTH = 800
HEIGHT = 600

log = open('rd_log.txt','w+')

#注释这个函数就会打印log到txt
#不影响print的效果
def print_to_log(*args, **kwargs): 
	pass
print_to_log('print log')


#数学库===========================================================================================================
def CMID(x, min, max):
	'''限制x的范围'''
	return min if x <= min else max if x >= max else x 

def interp(x1, x2, t):
	'''x1是表示点A，x2表示点B，t从0到1，表示从点A到B'''
	return x1 + (x2 - x1) * t

class vector:
	def __init__(self, x = 0, y = 0, z = 0, w = 1):
		self.x = x; self.y = y; self.z = z; self.w = w
		self.len = math.sqrt(x*x + y*y + z*z)
	def __repr__(self):
		return '( %.2f, %.2f, %.2f, %.2f)' %(self.x, self.y, self.z, self.w)
		#return '(' + str(self.x) + ',' +str(self.y)+','+str(self.z)+','+str(self.w) + ')'
	def dot_product(self, b):
		'''点乘，b为向量'''
		return self.x * b.x + self.y * b.y + self.z * b.z
	def cross_product(self, b):
		'''叉乘，b为向量'''
		return(vector(self.y * b.z - self.z * b.y, self.z * b.x - self.x * b.z, self.x * b.y - self.y * b.x))
	def normalize(self):
		'''向量单位化'''
		self.x /= self.len; self.y /= self.len; self.z /= self.len
		self.len = 1
	def add(a, b):
		return vector(a.x+b.x, a.y+b.y, a.z+b.z)
	def sub(a, b):
		return vector(a.x-b.x, a.y-b.y, a.z-b.z)

def v_interp(a:vector, b:vector, t:float):
	return vector(interp(a.x, b.x, t), interp(a.y, b.y, t), interp(a.z, b.z, t))

point = vector

class matrix:
	'''四阶矩阵，默认是单位矩阵'''
	def __init__(self):
		self.m = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
	def __repr__(self):
		s = ''
		m = self.m
		for i in range(4):
			s += '[%.2f %.2f %.2f %.2f]' %(m[i][0],m[i][1],m[i][2],m[i][3])
		return s
	def set_zero(self):
		self.m = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
	def set_identity(self):
		'''设为四阶单位矩阵'''
		self.m = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
	def set_translate(self, x, y, z):
		self.set_identity()
		self.m[3][0] = x
		self.m[3][1] = y
		self.m[3][2] = z
		print_to_log('set_translate',self)
	def set_scale(self, x, y, z):
		self.set_identity()
		self.m[0][0] = x
		self.m[1][1] = y
		self.m[2][2] = z
		print_to_log('set_scale',self)
	def set_rotate(self, x, y, z, theta):
		'''把自己变成一个表示旋转变化的矩阵，vector(x,y,z)旋转轴，theta旋转角'''
		qsin = math.sin(theta * 0.5)
		vec = vector(x,y,z)
		vec.normalize()
		x = vec.x * qsin
		y = vec.y * qsin
		z = vec.z * qsin
		w = math.cos(theta * 0.5)
		self.set_zero()
		self.m[0][0] = 1 - 2 * y * y - 2 * z * z 
		self.m[1][0] = 2 * x * y - 2 * w * z 
		self.m[2][0] = 2 * x * z + 2 * w * y 
		self.m[0][1] = 2 * x * y + 2 * w * z 
		self.m[1][1] = 1 - 2 * x * x - 2 * z * z 
		self.m[2][1] = 2 * y * z - 2 * w * x 
		self.m[0][2] = 2 * x * z - 2 * w * y 
		self.m[1][2] = 2 * y * z + 2 * w * x 
		self.m[2][2] = 1 - 2 * x * x - 2 * y * y 
		self.m[3][3] = 1
		#print_to_log('set_rotate',self, file = log)
	def set_lookat(self, eye:vector, at:vector, up:vector):
		'''设置摄像机，eyes是相机位置，at是物体位置'''
		zaxis = at.sub(eye)
		zaxis.normalize()
		xaxis = up.cross_product(zaxis)
		xaxis.normalize()
		yaxis = zaxis.cross_product(xaxis)
		self.set_zero()
		self.m[0][0] = xaxis.x; self.m[1][0] = xaxis.y; self.m[2][0] = xaxis.z; self.m[3][0] = -xaxis.dot_product(eye); 
		self.m[0][1] = yaxis.x; self.m[1][1] = yaxis.y; self.m[2][1] = yaxis.z; self.m[3][1] = -yaxis.dot_product(eye); 
		self.m[0][2] = zaxis.x; self.m[1][2] = zaxis.y; self.m[2][2] = zaxis.z; self.m[3][2] = -zaxis.dot_product(eye); 
		self.m[3][3] = 1
		#print_to_log('set_lookat',self, file = log)
	def set_perspective(self, fovy:float, aspect:float, znear:float, zfar:float):
		fax = 1 / math.tan(fovy * 0.5)
		self.set_zero()
		self.m[0][0] = fax / aspect
		self.m[1][1] = fax
		self.m[2][2] = zfar / (zfar - znear)
		self.m[3][2] = -znear * zfar / (zfar - znear)
		self.m[2][3] = 1
		#print_to_log('set_perspective',self, file = log)
	def add(self, b):
		'''矩阵加法，a,b都是4*4的list'''
		c = copy.deepcopy(self)
		for i in range(len(c.m)):
			for j in range(len(c.m[i])):
				c.m[i][j] += b.m[i][j]
		return c
	def sub(a,b):
		'''矩阵减法，a,b都是4*4的list'''
		c = copy.deepcopy(a)
		for i in range(len(c.m)):
			for j in range(len(c.m[i])):
				c.m[i][j] -= b.m[i][j]
		return c
	def mul(a, b):
		'''两个矩阵相乘，a,b都是4*4的list'''
		c = matrix()
		c.set_zero() #这里要注意，老子写漏了这一句查了一天的bug！
		for i in range(4):
			for j in range(4):
				for k in range(4):
					c.m[i][j] += a.m[i][k] * b.m[k][j]
		return c
	def scale(a,t):
		'''矩阵数乘，a都是4*4的list， t是数字'''
		c = copy.deepcopy(a)
		for i in range(len(c.m)):
			for j in range(len(c.m[i])):
				c.m[i][j] *= t
		return c
	def apply(self, v:vector):
		'''result = self * v'''
		result = vector(0,0,0,0)
		result.x = self.m[0][0] * v.x + self.m[1][0] * v.y + self.m[2][0] * v.z + self.m[3][0] * v.w
		result.y = self.m[0][1] * v.x + self.m[1][1] * v.y + self.m[2][1] * v.z + self.m[3][1] * v.w
		result.z = self.m[0][2] * v.x + self.m[1][2] * v.y + self.m[2][2] * v.z + self.m[3][2] * v.w
		result.w = self.m[0][3] * v.x + self.m[1][3] * v.y + self.m[2][3] * v.z + self.m[3][3] * v.w
		return result


#坐标变换===========================================================================================================
class transform:
	def __init__(self, w = WIDTH, h = HEIGHT):
		self.w = w #屏幕宽
		self.h = h #屏幕高
		self.world = matrix() #世界坐标变换，默认是单位矩阵
		self.view = matrix() #摄像机坐标变换，默认是单位矩阵
		self.proj = matrix() #projection 投影变换
		self.proj.set_perspective(math.pi * 0.5, w/h, 1, 500)
		self.update()
	def __repr__(self):
		return 'Transform\nWorld:' + str(self.world) +'\nview:'+ str(self.view) +'\nproj:'+ str(self.proj) +'\ntrans:'+ str(self.trans)
	def update(self):
		'''矩阵更新，计算 transform = world * view * projection'''
		self.trans = self.world.mul(self.view).mul(self.proj)
		#print_to_log(self, file = log)
	def apply(self, x:vector):
		'''将矢量 x 进行 project '''
		return self.trans.apply(x)
	def homogenize(self, a:vector):
		'''
		均质化，a 是一个在2*2*1长方体里面的vector，原点在中间
		把2*2*1投影到窗口大小，得到窗口坐标
		'''
		#print_to_log('transform.homogenize: a =', a, file = log)
		rhw = 1 / a.w #四元数的xyz除以w值就是真实坐标
		return vector((a.x * rhw + 1) * 0.5 * self.w, (1 - a.y * rhw) * 0.5 * self.h, a.z* rhw, 1)
		'''
		x真实坐标+1：因为窗口坐标跟我们的坐标都是从左到右增加的
		1 - y的真实坐标：我们的坐标是从下到上增加的，但是窗口坐标相反
		也就是我们的原点在屏幕左下角，窗口坐标的原点在屏幕左上角
		为什么要乘0.5呢，因为长方方体的xy边长是2啊，z边长是1所以不用乘
		为什么当窗口长宽比不同的时候画出来不会形变呢，因为cvv也是按照窗口比例的
		'''

def trans_check_cvv(v:vector):
	'''
	检查齐次坐标同 cvv 的边界用于视锥裁剪
	规则观察体（CVV, Canonical View Volume）
	'''
	w = v.w
	print_to_log('trans_check_cvv, w=', w, file = log)
	check = 0
	if v.z < 0: check |= 1
	if v.z > w: check |= 2
	if v.x < -w: check |= 4
	if v.x > w: check |= 8
	if v.y < -w: check |= 16
	if v.x > w: check |= 32
	print_to_log('check =', check, file = log)
	return check


#几何计算===========================================================================================================
class color:
	def __init__(self, r:float = 0, g:float= 0, b:float = 0):
		self.r = r; self.g = g; self.b = b
	def __repr__(self):
		return '(%.2f, %.2f, %.2f)' %(self.r,self.g, self.b)
	def add(self, c):
		self.r += c.r; self.g += c.g; self.b += c.b #渐变背景要用到

class texcoord:
	def __init__(self, u:float = 0, v:float = 0):
		self.u = u; self.v = v
	def __repr__(self):
		return '(%.2f, %.2f)' %(self.u,self.v)

class vertex:
	def __init__(self, pos:point = None, tc:texcoord = None, c:color = None):
		if pos == None:
			self.pos = point()
		else:
			self.pos = pos
		if tc == None:
			self.tc = texcoord()
		else:
			self.tc = tc
		if c == None:
			self.c = color()
		else:
			self.c = c 
		'''
		这里有个大坑，不能在默认参数里直接实例化某个类（比如pos = point()），
		不然所有使用了默认参数的对象的本属性(pos)会指向同一个实例(point())
		'''
		self.vertex_rhw_init
	def vertex_rhw_init(self):
		self.rhw = 1 / self.pos.w
		self.tc.u *= self.rhw; self.tc.v *= self.rhw
		self.c.r *= self.rhw; self.c.g *= self.rhw; self.c.b *= self.rhw
	def __repr__(self):
		return 'pos:' + str(self.pos) + 'tc:' + str(self.tc) + 'c:' + str(self.c)
	def vertex_interp(self, x1, x2, t:float):
		'''x1 x2 都是vertex'''
		self.pos = v_interp(x1.pos, x2.pos, t)
		self.tc.u = interp(x1.tc.u, x2.tc.u, t)
		self.tc.v = interp(x1.tc.v, x2.tc.v, t)
		self.c.r = interp(x1.c.r, x2.c.r, t)
		self.c.g = interp(x1.c.g, x2.c.g, t)
		self.c.b = interp(x1.c.b, x2.c.b, t)
		self.rhw = interp(x1.rhw, x2.rhw, t)
		print_to_log('vertex_interp:x1:', x1.pos, ' x2:', x2.pos,' v:', self.pos, file = log)
	def division(self, x1, x2, w:int):
		'''x1 x2 都是vertex，w表示width
		把从x1到x2的向量分成w份，每份作为scanline的step'''
		inv = 1 / w
		self.pos.x = (x2.pos.x - x1.pos.x) * inv
		self.pos.y = (x2.pos.y - x1.pos.y) * inv
		self.pos.z = (x2.pos.z - x1.pos.z) * inv
		self.pos.w = (x2.pos.w - x1.pos.w) * inv
		self.tc.u = (x2.tc.u - x1.tc.u) * inv
		self.tc.v = (x2.tc.v - x1.tc.v) * inv
		self.c.r = (x2.c.r - x1.c.r) * inv
		self.c.g = (x2.c.g - x1.c.g) * inv
		self.c.b = (x2.c.b - x1.c.b) * inv
		self.rhw = (x2.rhw - x1.rhw) * inv
	def add(self, a):
		#print_to_log('vertex.add', self.pos, a.pos, self.rhw, a.rhw, file = log)
		self.pos.x += a.pos.x; self.pos.y += a.pos.y; self.pos.z += a.pos.z; self.pos.w += a.pos.w
		self.rhw += a.rhw
		self.tc.u += a.tc.u; self.tc.v += a.tc.v
		self.c.r += a.c.r; self.c.g += a.c.g; self.c.b += a.c.b

mesh = []
mesh.append(vertex(point(-1,-1,1,1), texcoord(0,0), color(1, 0.2, 0.2)))
mesh.append(vertex(point(1,-1,1,1), texcoord(0,1), color(0.2, 1, 0.2)))
mesh.append(vertex(point(1,1,1,1), texcoord(1,1), color(0.2, 0.2, 1)))
mesh.append(vertex(point(-1,1,1,1), texcoord(1,0), color(1, 0.2, 1)))
mesh.append(vertex(point(-1,-1,-1,1), texcoord(0,0), color(1, 1, 0.2)))
mesh.append(vertex(point(1,-1,-1,1), texcoord(0,1), color(0.2, 1, 1)))
mesh.append(vertex(point(1,1,-1,1), texcoord(1,1), color(1, 0.3, 0.3)))
mesh.append(vertex(point(-1,1,-1,1), texcoord(1,0), color(0.2, 1, 0.3)))

class edge:
	def __init__(self, v1:vertex, v2:vertex):
		self.v1 = v1; self.v2 = v2
		self.v = vertex(pos = point(0,0,0,1), tc = texcoord(), c = color())
	def __repr__(self):
		return 'v1:' + str(self.v1) +',v2:' + str(self.v2) + ',v:' + str(self.v)

class scanline():
	def __init__(self, step:vertex = None, x:int = 0, y:int = 0, w:int = 0):
		#还有一个v属性在init_scanline里面添加
		if step == None:
			self.step = vertex()
		else:
			self.step = step
		self.x = x; self.y = y; self.w = w
	def __repr__(self):
		return 'v:' + str(self.v) + 'step:' + str(self.step) + 'x:' + str(self.x)+ 'y:' + str(self.y)+ 'w:' + str(self.w)

class trapezoid:
	'''不规则四边形，梯形'''
	def __init__(self, top:float, bottom:float, left:edge, right:edge):
		self.top = top; self.bottom = bottom; self.left = left; self.right = right
	def __repr__(self):
		return 'top:%.2f, bottom:%.2f '%(self.top, self.bottom)+ '\nleft:' + str(self.left)+ '\nright:' + str(self.right) +'\n'
	def edge_interp(self, y:float):
		'''按照 Y 坐标计算出左右两条边纵坐标等于 Y 的顶点'''
		s1 = self.left.v2.pos.y - self.left.v1.pos.y
		s2 = self.right.v2.pos.y - self.right.v1.pos.y
		t1 = (y - self.left.v1.pos.y) / s1
		t2 = (y - self.right.v1.pos.y) / s2
		self.left.v.vertex_interp(self.left.v1, self.left.v2, t1)
		print_to_log('self.left.v:', self.left.v, file = log)
		self.right.v.vertex_interp(self.right.v1, self.right.v2, t2)
		print_to_log('self.right.v:', self.right.v, file = log)
	def init_scanline(self, sl:scanline, y:int):
		'''根据左右两边的端点，初始化计算出扫描线的起点和步长'''
		print_to_log('\ninit_scanline, y =', y, file = log)
		print_to_log('\n', self, file = log)
		width = self.right.v.pos.x - self.left.v.pos.x
		sl.x = int(self.left.v.pos.x + 0.5)
		sl.w = int(self.right.v.pos.x + 0.5) - sl.x
		sl.y = y
		sl.v = copy.deepcopy(self.left.v)
		if(self.left.v.pos.x >= self.right.v.pos.x):
			sl.w = 0
		sl.step.division(self.left.v, self.right.v, width)
		#这里如果width == 0的话会报错，但是因为执行本函数之前先执行了edge_interp
		#所以左右的v就不一样了


def trapezoid_init_triangle(p1:vertex, p2:vertex, p3:vertex):
	'''根据三角形生成 0-2 个梯形，并且返回合法梯形'''
	result = []
	print_to_log('trapezoid_init_triangle',p1,p2,p3, file = log)
	if p1.pos.y == p2.pos.y == p3.pos.y or p1.pos.x == p2.pos.x == p3.pos.x:
		print_to_log('result0:', result, file = log)
		return result
	if p1.pos.y > p2.pos.y: p1,p2 = p2,p1 #p123.pos.y从小到大排序
	if p1.pos.y > p3.pos.y: p1,p3 = p3,p1
	if p2.pos.y > p3.pos.y: p3,p2 = p2,p3
	
	if p1.pos.y == p2.pos.y: #triangle down
		if p1.pos.x > p2.pos.x: p1,p2 = p2,p1
		trap = trapezoid(p1.pos.y, p3.pos.y, edge(p1, p3), edge(p2, p3))
		result.append(trap)
		print_to_log('result1:', result, file = log)
		return result

	if p2.pos.y == p3.pos.y: #triangle up
		if p2.pos.x > p3.pos.x: p3,p2 = p2,p3
		trap = trapezoid(p1.pos.y, p3.pos.y, edge(p1, p2), edge(p1, p3))
		result.append(trap)
		print_to_log('result2:', result, file = log)
		return result

	#k = (p3.pos.y - p1.pos.y) / (p2.pos.y - p1.pos.y)
	#x = p1.pos.x - (p1.pos.x - p3.pos.x) * k

	k = (p3.pos.y - p1.pos.y) / (p2.pos.y - p1.pos.y);
	x = p1.pos.x + (p2.pos.x - p1.pos.x) * k;

	#if(p2.pos.x <= x): #triangle left
	if(x <= p3.pos.x): #triangle left
		trap = trapezoid(p1.pos.y, p2.pos.y, edge(p1,p2), edge(p1,p3))
		result.append(trap)
		trap = trapezoid(p2.pos.y, p3.pos.y, edge(p2,p3), edge(p1,p3))
		result.append(trap)
	else: #triangle right
		trap = trapezoid(p1.pos.y, p2.pos.y, edge(p1,p3), edge(p1,p2))
		result.append(trap)
		trap = trapezoid(p2.pos.y, p3.pos.y, edge(p1,p3), edge(p2,p3))
		result.append(trap)
	print_to_log('result3:', result, file = log)
	return result

#渲染设备===========================================================================================================
RENDER_STATE_WIREFRAME = 1
RENDER_STATE_TEXTURE = 2
RENDER_STATE_COLOR = 4

class decive:
	def __init__(self, width = WIDTH, height = HEIGHT):
		self.transform = transform(width, height)      # 坐标变换器
		self.width = width; self.height = height

		self.render_state = RENDER_STATE_TEXTURE   #渲染状态
		self.background = [color(200,200,200), color(20,20,20)]   #背景颜色,从上往下渐变
		self.foreground = color(255,255,255)         #线框颜色

		self.init_buf() #初始化深度缓存、背景缓存和图像缓存
		self.set_texture()  #纹理缓存
	def set_texture(self):
		self.texture = Image.open('texture.bmp')
		self.tex_width = self.texture._size[0]
		self.tex_height = self.texture._size[1]
		assert self.tex_width <= 1024 and self.tex_height <= 1024
		self.max_u = self.tex_width - 1
		self.max_v = self.tex_height - 1
	def init_buf(self, mode:int = 1):
		#mode == 0/1: 纯色背景/渐变背景
		self.zbuffer = [[0,]*self.width for i in range(self.height)] #深度缓存：zbuffer[y] 为第 y行指针
		self.bgbuffer = Image.new("RGB",(WIDTH,HEIGHT)) #背景缓存，不用每次计算背景，效率高一点
		if mode == 1:
			r = (self.background[0].r - self.background[1].r) / (self.height - 1)
			g = (self.background[0].g - self.background[1].g) / (self.height - 1)
			b = (self.background[0].b - self.background[1].b) / (self.height - 1)
			c_step = color(-r,-g,-b)
			cc = copy.deepcopy(self.background[0])
			for j in range(self.height):
				for i in range(self.width):
					self.bgbuffer.putpixel([i,j],(int(cc.r), int(cc.g), int(cc.b)))
				cc.add(c_step)
		else: #mode == 0
			cc = self.background[1]
			for j in range(self.height):
				for i in range(self.width):
					self.bgbuffer.putpixel([i,j],(int(cc.r), int(cc.g), int(cc.b)))
		self.fbuffer = copy.deepcopy(self.bgbuffer)			
	def clear_buf(self):
		self.zbuffer = [[0,]*self.width for i in range(self.height)] #深度缓存清空
		self.fbuffer = copy.deepcopy(self.bgbuffer)
	def pixel(self, x, y, color):
		'''画点'''
		if x >= 0 and x < self.width and y >= 0 and y < self.height:
			x = int(x)
			y = int(y)
			self.fbuffer.putpixel([x,y],(color.r,color.g,color.b))
			#print_to_log('put', x, y, ':', color.r, color.g, color.b)
	def line(self, x1, y1, x2, y2, c = color(255,255,255)):
		'''画线'''
		if x1 == x2 and y1 == y2:
			self.pixel(x1, y1, c)
		elif x1 == x2:
			if y2 < y1: 
				y1, y2 = y2, y1
			for i in range(y1, y2 + 1):
				self.pixel(x1, i, c)
		elif y1 == y2:
			if x2 < x1: 
				x1, x2 = x2, x1
			for i in range(x1, x2 + 1):
				self.pixel(i, y1, c)
		else:
			if x2 < x1: 
				x1, x2 , y1, y2 = x2, x1, y2, y1
			x, y = x1, y1
			a, b = abs(y2 - y1), x2 - x1
			k = abs(y2 - y1) / (y2 - y1)
			f = 0
			for i in range(x2 - x1 + abs(y2 - y1)):
				self.pixel(x,y,c)
				if f >= 0:
					f -= a
					x += 1
				else:
					f += b
					y += k
			self.pixel(x2, y2, c)
	def texture_read(self, u:float, v:float):
		u = u * self.max_u
		v = v * self.max_v
		x = int(u + 0.5)
		y = int(v + 0.5)
		x = CMID(x, 0, self.tex_width - 1)
		y = CMID(y, 0, self.tex_height - 1)
		return self.texture.getpixel((x,y))
	#渲染实现==========================================================
	def draw_scanline(self, sl:scanline):
		'''绘制扫描线'''
		y = sl.y
		x = sl.x
		w = sl.w
		zb = self.zbuffer[y]
		width = self.width
		render_state = self.render_state
		for i in range(w):
			if x >= 0 and x < width:
				rhw = sl.v.rhw
				#print_to_log('x = %d, y = %d, rhw = %f, zb[x] = %f'%(x,y, rhw, zb[x]), file = log)
				if(rhw >= zb[x]):
					w = 1 / rhw
					zb[x] = rhw
					print_to_log('ZBUF: zb[x] = %.2f' %(rhw), file = log)
					if render_state & RENDER_STATE_COLOR:
						R = CMID(int(sl.v.c.r * w * 255), 0, 255)
						G = CMID(int(sl.v.c.g * w * 255), 0, 255)
						B = CMID(int(sl.v.c.b * w * 255), 0, 255)
						self.fbuffer.putpixel((x,y),(R,G,B))
					if render_state & RENDER_STATE_TEXTURE:
						u = sl.v.tc.u * w
						v = sl.v.tc.v * w
						self.fbuffer.putpixel((x,y),self.texture_read(u,v))
			sl.v.add(sl.step)
			if x >= width: break
			x += 1
	def render_trap(self, trap:trapezoid):
		'''主渲染函数'''
		print_to_log('render_trap', file = log)
		sl = scanline()
		top = int(trap.top + 0.5)
		bottom = int(trap.bottom + 0.5)
		for j in range(top, bottom):
			if j >= 0 and j < self.height:
				trap.edge_interp(j + 0.5)
				trap.init_scanline(sl, j)
				self.draw_scanline(sl)
			if j >= self.height:
				break
	def draw_primitive(self, v1:vertex, v2:vertex, v3:vertex):
		'''根据render_state绘制原始三角形'''
		print_to_log('draw_primitive:', file = log)
		print_to_log(v1, v2, v3, sep = '\n', file = log)

		c1 = self.transform.apply(v1.pos) #按照 Transform 变化
		c2 = self.transform.apply(v2.pos)
		c3 = self.transform.apply(v3.pos)
		print_to_log('draw_primitive(c):', file = log)
		print_to_log(c1, c2, c3, sep = '\n', file = log)

		#裁剪，注意此处可以完善为具体判断几个点在 cvv内以及同cvv相交平面的坐标比例
		#进行进一步精细裁剪，将一个分解为几个完全处在 cvv内的三角形
		if not trans_check_cvv(c1) == 0 or not trans_check_cvv(c2) == 0 or not trans_check_cvv(c3) == 0:
			return

		p1 = self.transform.homogenize(c1) #单位化
		p2 = self.transform.homogenize(c2)
		p3 = self.transform.homogenize(c3)
		print_to_log('draw_primitive(p):', file = log)
		print_to_log(p1, p2, p3, sep = '\n', file = log)


		#纹理或色彩绘制
		if self.render_state & (RENDER_STATE_TEXTURE | RENDER_STATE_COLOR):
			#背面消除，先计算法线
			a = p2.sub(p1)
			b = p3.sub(p2)
			p = a.cross_product(b) #法线
			if p.dot_product(vector(0,0,1)) < 0:
				return
			#print_to_log('纹理或色彩绘制, render_state =',self.render_state, file = log)
			t1 = vertex(p1, texcoord(v1.tc.u, v1.tc.v), color(v1.c.r, v1.c.g, v1.c.b))
			t2 = vertex(p2, texcoord(v2.tc.u, v2.tc.v), color(v2.c.r, v2.c.g, v2.c.b))
			t3 = vertex(p3, texcoord(v3.tc.u, v3.tc.v), color(v3.c.r, v3.c.g, v3.c.b))
			t1.pos.w = c1.w
			t2.pos.w = c2.w
			t3.pos.w = c3.w
			t1.vertex_rhw_init()
			t2.vertex_rhw_init()
			t3.vertex_rhw_init()
			print_to_log('t1.rhw:%f, t2.rhw:%f, t3.rhw:%f'%(t1.rhw,t2.rhw,t3.rhw), file = log)
			traps = trapezoid_init_triangle(t1, t2, t3)#拆分三角形
			print_to_log('traps:',traps, file = log)
			if len(traps) >= 1:
				self.render_trap(traps[0])
			if len(traps) >= 2:
				self.render_trap(traps[1])
		#线框绘制
		if self.render_state & RENDER_STATE_WIREFRAME: 
			self.line(int(p1.x), int(p1.y), int(p2.x), int(p2.y), self.foreground)
			self.line(int(p1.x), int(p1.y), int(p3.x), int(p3.y), self.foreground)
			self.line(int(p3.x), int(p3.y), int(p2.x), int(p2.y), self.foreground)
	def draw_plane(self, a:int, b:int, c:int, d:int):
		p1 = mesh[a]; p2 = mesh[b]; p3 = mesh[c]; p4 = mesh[d]
		p1.tc.u, p1.tc.v = 0,0
		p2.tc.u, p2.tc.v = 1,0
		p3.tc.u, p3.tc.v = 1,1
		p4.tc.u, p4.tc.v = 0,1
		self.draw_primitive(p1,p2,p3)
		self.draw_primitive(p3,p4,p1)
	def draw_box(self, theta:float):
		m = matrix()
		m.set_rotate(-1,-0.5,1,theta)
		self.transform.world = m
		self.transform.update()
		self.draw_plane(0, 1, 2, 3)
		self.draw_plane(7, 6, 5, 4)
		self.draw_plane(0, 4, 5, 1)
		self.draw_plane(1, 5, 6, 2)
		self.draw_plane(2, 6, 7, 3)
		self.draw_plane(3, 7, 4, 0)
	def camera_at_zero(self, x:float, y:float, z:float):
		eye = point(x,y,z,1)
		at = point(0,0,0,1)
		up = point(0,0,1,1)
		self.transform.view.set_lookat(eye, at, up)
		self.transform.update()

class Draw:
	def __init__(self, master, width = WIDTH, height = HEIGHT):
		self.master = master
		self.width = width
		self.height = height
		self.cv = Canvas(self.master, width = self.width, height = self.height, background = 'lightblue')
		self.cv.focus_set()
		self.cv.pack()

		self.cv.bind('<Left>', self.turn_left)
		self.cv.bind('<Right>', self.turn_right)
		self.cv.bind('<Up>', self.go_near)
		self.cv.bind('<Down>', self.go_far)
		self.cv.bind('<space>', self.change_mode)

		self.dc = decive(self.width, self.height)
		self.states = [RENDER_STATE_TEXTURE, RENDER_STATE_COLOR, RENDER_STATE_WIREFRAME]
		self.cv.ptbg = ImageTk.PhotoImage(self.dc.bgbuffer)
		self.cv.bg = self.cv.create_image(0,0,image = self.cv.ptbg, anchor = 'nw')
		self.indicator = 0
		self.alpha = 1
		self.pos = 3.5
		self.dc.camera_at_zero(self.pos,0,0)
		self.now_img = -1
		self.render()


	def render(self):
		self.dc.clear_buf()
		if not self.now_img == -1:
			self.cv.delete(self.now_img)

		self.dc.draw_box(self.alpha)  #0.93s，卡点

		self.cv.ptimg = ImageTk.PhotoImage(self.dc.fbuffer)  #0.016s，不能省
		self.now_img = self.cv.create_image(0,0,image = self.cv.ptimg, anchor = 'nw')  #0.0s

	def turn_left(self, event):
		self.alpha += 0.05
		self.render()
	def turn_right(self, event):
		self.alpha -= 0.05
		self.render()
	def go_near(self, event):
		self.pos += 0.5
		self.dc.camera_at_zero(self.pos,0,0)
		self.render()
	def go_far(self, event):
		self.pos -= 0.5
		self.dc.camera_at_zero(self.pos,0,0)
		self.render()
	def change_mode(self, event):
		self.indicator = (self.indicator + 1) % 3
		self.dc.render_state = self.states[self.indicator]
		self.render()

root = Tk()
root.geometry(str(WIDTH) + 'x' + str(HEIGHT))
Draw(root, WIDTH, HEIGHT)
root.mainloop()
log.close()