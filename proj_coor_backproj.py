# from simple_renderer import SimpleRenderer
import math
from math import pi
from euler import eulerAnglesToRotationMatrix
import cv2
import numpy as np
import inout, renderer, misc
# import matplotlib.pyplot as plt
# from sys import platform as sys_pf
# if sys_pf == 'darwin':
#     import matplotlib
#     matplotlib.use("TkAgg")
import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
    print(matplotlib.get_backend())
    from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



class SimpleRenderer:
	def __init__(self, model_path):
		self.model = inout.load_ply(model_path)
		self.K = np.array([
			[679.78228891, 0.0,  316.4596997],
			[0.0, 678.69975492,  245.26957305],
			[0.0, 0.0, 1.0]],dtype=np.float32)

	def render(self, R, t):
		ren, canvas = renderer.render(self.model, (960, 720), self.K, R, t,surf_color=(1,1,1,1), bg_color=(0, 0, 0, 0), mode='rgb+depth')
		self.canvas = canvas
		ren_rgb, ren_depth = ren
		yz_flip = np.eye(4, dtype=np.float32)
		yz_flip[1, 1], yz_flip[2, 2] = -1, -1

		mat_world2camera = np.dot(yz_flip, canvas.mat_view.T)
		mat_camera2world = np.linalg.inv(mat_world2camera)

		mat_rotation = mat_world2camera[:3, :3]
		vec_translation = mat_world2camera[:3, 3].reshape([3, 1])
		return ren_rgb, ren_depth, mat_rotation, vec_translation
	
	def render_proj_2d_img(self):
		simple_renderer = self.model
		i_front = 0.0
		i_side_right = 89.0 
		i_side_left = 269.0
		# i_back = 179.0
		i_top = 269.0
		# i_bottom = 89.0
		four_view_angles = {1:["front"], 2:["side_right"], 3:["side_left"], 4:["top"]}
		eulerangles = [[179.0 * pi / 180.0, i_front * pi / 180.0, 0], [179.0 * pi / 180.0, i_side_right * pi / 180.0, 0],
						[179.0 * pi / 180.0, i_side_left * pi / 180.0, 0], [i_top * pi / 180.0, 0, 0]]

		t = np.array([
			[0, 0, 600]
		], dtype=np.float32)

		for idx, eulerangle in enumerate(eulerangles):
			rotation = eulerAnglesToRotationMatrix(eulerangle)
			R = np.array(rotation,dtype=np.float32)
			ren_rgb, ren_depth, mat_rotation, vec_translation = self.render(R, t)
			rgb_path = '/Users/luchenliu/Desktop/Intern/1_projection/test_images/four_views/%s.jpg'%(four_view_angles[idx+1][0])
			depth_path = '/Users/luchenliu/Desktop/Intern/1_projection/test_images/four_views/%s_depth.jpg'%(four_view_angles[idx+1][0])
			cv2.imwrite(rgb_path, ren_rgb)
			cv2.imwrite(depth_path, ren_depth)

			depth_file_path = "/Users/luchenliu/Desktop/Intern/1_projection/test_images/four_views/%s_depth_info.txt"%(four_view_angles[idx+1][0])
			depth_file = open(depth_file_path, 'w')
			for r in range (0,ren_depth.shape[0]):
				for c in range (0,ren_depth.shape[0]):
					depth_file.write('%04f%s'%(ren_depth[r,c]," "))
			depth_file.close()

			mask_img_path = '/Users/luchenliu/Desktop/Intern/1_projection/test_images/four_views/%s_mask.jpg'%(four_view_angles[idx+1][0])
			four_view_angles[idx+1].append(ren_rgb)
			four_view_angles[idx+1].append(ren_depth)
			four_view_angles[idx+1].append(mat_rotation)
			four_view_angles[idx+1].append(vec_translation)
			four_view_angles[idx+1].append(mask_img_path)

		return four_view_angles


def find_contour(img):
	all_contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	if len(all_contours) != 0:
		all_sorted_cnt = sorted(all_contours, key=cv2.contourArea)[-1:]
		largest_cnt = all_sorted_cnt[0]
	else:
		largest_cnt = None
		all_sorted_cnt = None
		print("no contours")
	return largest_cnt, all_sorted_cnt


def draw_contour(img, canvas, contour_thickness):
	largest_cnt, all_sorted_cnt = find_contour(img)
	if all_sorted_cnt is not None:
		for i, largest_cnt in enumerate(all_sorted_cnt):
			cv2.drawContours(canvas, largest_cnt, -1, [0, 255, 0], contour_thickness)
	return largest_cnt, all_sorted_cnt


def get_most_pts(all_sorted_cnt):
	cnt = all_sorted_cnt[0]
	leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
	rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
	topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
	bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
	return leftmost, rightmost, topmost, bottommost


def addImage(projected_img, mask_img):
	alpha = 0.7
	beta = 1 - alpha
	gamma = 0
	overlay_img = np.asarray(cv2.addWeighted(projected_img, alpha, mask_img, beta, gamma))
	return overlay_img
        

def manual_click(event, x, y, flags, param):
	global mouseX, mouseY
	# global deformation_key_points 
	overlay_img = param[0]
	model_depth_img = param[1]
	deformation_key_points = param[2]
	if event == cv2.EVENT_LBUTTONDBLCLK:
		mouseX, mouseY = x, y
		print(mouseY, mouseX, model_depth_img[mouseY,mouseX])
		show_points_pos(overlay_img, mouseX, mouseY)
		deformation_key_points.append([mouseY, mouseX])
	if event == cv2.EVENT_RBUTTONDBLCLK:
		return

def show_points_pos(overlay_img, mouseX, mouseY):
	overlay_img[mouseY - 3:mouseY + 3, mouseX - 3:mouseX + 3] = [0, 128, 255]
	cv2.imshow('overlay_img', overlay_img)


def prepare_point_coors(point_list):
	if not point_list:
		return 
	if len(point_list)%2 != 0:
		print ("Expect point list with an even number, but an odd number is provided. Please select the points again!")
		return 
	target_start_idx = len(point_list)//2
	projected_key_points = point_list[:target_start_idx]
	mask_key_points = point_list[target_start_idx:]
	return projected_key_points, mask_key_points


def get_transform_matrix(depth_img, projected_key_points,mask_key_points,mat_rotation,vec_translation):
	if not projected_key_points:
		return None
	if not mask_key_points:
		return None
	world_3d_key_points = []
	target_world_3d_key_points = []
	
	print("length of projected:",len(projected_key_points))
	print("length of mask",len(mask_key_points))
	count = 0
	for projected_key_point in projected_key_points:
		print("projected_key_point:",projected_key_point)
		u = projected_key_point[0]
		v = projected_key_point[1]
		w = depth_img[u, v]
		if w == 0.0:
			continue
		print("depth:",w)
		camera_point = np.array([v, u, w], dtype=np.float32).reshape(3, 1)
		world_point = np.dot(mat_rotation.T, camera_point - vec_translation)
		# print("world_point",world_point)
		x = world_point[0]
		y = world_point[1]
		z = world_point[2]
		world_3d_key_points.append([x, y, z])
		count += 1
	world_3d_key_points = np.array(world_3d_key_points,dtype=np.float32).reshape(-1, 3)
	print("count:",count)
	print("world_3d_key_points",world_3d_key_points)

	count2 = 0
	for mask_key_point in mask_key_points:
		print("mask_key_point:",mask_key_point)
		m = mask_key_point[0]
		n = mask_key_point[1]
		d = w
		print("depth:",w)
		target_camera_point = np.array([n, m, d], dtype=np.float32).reshape(3, 1)
		target_world_point = np.dot(mat_rotation.T, camera_point - vec_translation)
		# print(target_world_point)
		a = target_world_point[0]
		b = target_world_point[1]
		c = target_world_point[2]
		target_world_3d_key_points.append([a, b, c])
		count2 += 1		
	target_world_3d_key_points = np.array(target_world_3d_key_points,dtype=np.float32).reshape(-1, 3)
	print("count2:",count2)
	print("target_world_3d_key_points",target_world_3d_key_points)

	return world_3d_key_points, target_world_3d_key_points



def main():
	simple_renderer = SimpleRenderer('glasses.ply')
	four_view_angles = simple_renderer.render_proj_2d_img()

	for view_idx in range (1,5):
		rotation_mat_3by3 = np.array(four_view_angles[view_idx][3])
		translation_vec_1by3 = np.array(four_view_angles[view_idx][4])
		mask_img_path = four_view_angles[view_idx][5]

		model_rgb_img = np.array(four_view_angles[view_idx][1])
		model_depth_img = np.array(four_view_angles[view_idx][2])
		print("depth value:",model_depth_img[200,255])
		model_gray_img = cv2.cvtColor(model_rgb_img, cv2.COLOR_BGR2GRAY)

		mask_rgb_img = cv2.imread(mask_img_path)
		mask_gray_img = cv2.cvtColor(mask_rgb_img, cv2.COLOR_BGR2GRAY);

		canvas_size = mask_rgb_img.shape
		blank_canvas = np.zeros(canvas_size, dtype=np.uint8)
		ret, thresh = cv2.threshold(mask_gray_img, 210, 255, 0, cv2.THRESH_BINARY)
		thresh_cvt = 255 - thresh

		projected_largest_cnt, projected_all_sorted_cnt = find_contour(model_gray_img)	
		projected_leftmost, projected_rightmost, projected_topmost, projected_bottommost = get_most_pts(projected_all_sorted_cnt)
		# print("projected_topmost[0]",projected_topmost)
		# print("projected_bottommost[0]",projected_bottommost)
		# print("projected_leftmost[1]",projected_leftmost)
		# print("projected_rightmost[1]",projected_rightmost)

		mask_largest_cnt, mask_all_sorted_cnt = draw_contour(thresh_cvt, blank_canvas, 2)
		mask_leftmost, mask_rightmost, mask_topmost, mask_bottommost = get_most_pts(mask_all_sorted_cnt)
		
		# cropout projected img JUST TO GET THE SIZE
		height_model_whole, length_model_whole, _ = model_rgb_img.shape
		print("model_rgb_img shape:",model_rgb_img.shape)

		cropped_model_image = model_rgb_img[projected_topmost[1]-5:projected_bottommost[1]+5, projected_leftmost[0]-5:projected_rightmost[0]+5]
		height_model, length_model, _ = cropped_model_image.shape
		print("cropped_model_image shape:",cropped_model_image.shape)

		# cropout mask by bounding box
		cropped_mask_gray_img = mask_gray_img[mask_topmost[1]-5:mask_bottommost[1]+5, mask_leftmost[0]-5:mask_rightmost[0]+5]
		cropped_mask_rgb_img = mask_rgb_img[mask_topmost[1]-5:mask_bottommost[1]+5, mask_leftmost[0]-5:mask_rightmost[0]+5]
		print("cropped_mask_rgb_img shape:",cropped_mask_rgb_img.shape)

		# resize ONLY the cropped part of mask
		mask_glasses_resized = cv2.resize(cropped_mask_rgb_img, (length_model, height_model), interpolation=cv2.INTER_LINEAR)
		print("mask_glasses_resized shape:",mask_glasses_resized.shape)

		# cv2.namedWindow('mask_glasses_resized')
		# cv2.imshow('mask_glasses_resized', mask_glasses_resized)

		# adding other parts to form the resized-mask image
		top_border = projected_topmost[1]-5
		bottom_border = height_model_whole - (projected_bottommost[1]+5)
		left_border = projected_leftmost[0]-5
		right_border = length_model_whole - (projected_rightmost[0]+5)


		mask_glasses_resized_enlarged = cv2.copyMakeBorder(mask_glasses_resized,top_border,bottom_border,left_border,right_border, cv2.BORDER_CONSTANT,value=[255,255,255]) 
		
		# check size again
		print("size should be same:",model_rgb_img.shape, mask_glasses_resized_enlarged.shape)

		_, _ = draw_contour(model_gray_img, model_rgb_img, 2)
		overlay_img = addImage(model_rgb_img, mask_glasses_resized_enlarged)

		# model_rgb_img = np.array(model_rgb_img,dtype = np.float32)
		# mask_rgb_img_resized = np.array(mask_rgb_img_resized,dtype = np.float32)

		deformation_key_points = []
		cv2.namedWindow('overlay_img')
		cv2.setMouseCallback('overlay_img', manual_click, [overlay_img,model_depth_img,deformation_key_points])
		cv2.imshow('overlay_img', overlay_img)
		if (cv2.waitKey() != "q"):
			cv2.waitKey(0)

	    # SELECT KEYPOINTS
		print(deformation_key_points)
		projected_key_points, mask_key_points = prepare_point_coors(deformation_key_points)
		print("projected_key_points: ",projected_key_points)
		print("mask_key_points: ",mask_key_points)
		world_3d_key_points, target_world_3d_key_points = get_transform_matrix(model_depth_img,projected_key_points,mask_key_points,rotation_mat_3by3,translation_vec_1by3)

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		for i in range(world_3d_key_points.shape[0]):
			x = world_3d_key_points[i, 0]
			y = world_3d_key_points[i, 1]
			z = world_3d_key_points[i, 2]
			ax.scatter(x, y, z, c='r', marker='o')
		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')
		plt.show()


if __name__ == '__main__':
	main()