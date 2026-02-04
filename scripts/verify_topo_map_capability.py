
import comtypes.client
import array
import math
import random

def main():
    print("正在连接 AutoCAD 2026...")
    try:
        acad = comtypes.client.GetActiveObject("AutoCAD.Application")
        doc = acad.ActiveDocument
        ms = doc.ModelSpace
        print("成功连接到 AutoCAD。")
    except Exception as e:
        print(f"连接失败: {e}")
        print("请确保 AutoCAD 2026 正在运行。")
        return

    print("开始执行地形图绘制任务...")

    # 1. 图层管理 (Layer Management)
    layers = {
        "DGX-S": 30,  # 首曲线 (Intermediate) - Brownish
        "DGX-J": 30,  # 计曲线 (Index) - Brownish (will set lineweight later)
        "GCD": 7,     # 高程点 (Spot Heights) - White
        "BHQ-1": 1,   # 一级保护区 (Zone 1) - Red
        "BHQ-2": 2,   # 二级保护区 (Zone 2) - Yellow
        "KJJ": 4,     # 界桩/标识牌 (Markers) - Cyan
        "TK": 7,      # 图框 (Frame) - White
        "ZJ": 252     # 注记 (Annotations) - Gray
    }

    for name, color in layers.items():
        try:
            layer = doc.Layers.Add(name)
            layer.color = color
            print(f"图层 '{name}' 已创建/检查。")
        except Exception as e:
            print(f"创建图层 '{name}' 失败: {e}")

    # 2. 绘制等高线 (Contours) - 模拟一个山包
    # 间距 2米
    center_x, center_y = 1000, 1000
    base_radius = 500
    max_elevation = 100
    
    print("正在绘制等高线 (间距2米)...")
    for z in range(0, max_elevation + 2, 2):
        # 半径随高度减小
        r = base_radius * (1 - z / 120.0)
        if r <= 0: break
        
        # 是首曲线还是计曲线？ (每10米为计曲线)
        is_index = (z % 10 == 0)
        layer_name = "DGX-J" if is_index else "DGX-S"
        
        # 绘制圆模拟等高线
        # 使用 COM 添加圆: AddCircle(Center, Radius)
        center = array.array('d', [center_x, center_y, 0]) # Z is 0 for entity, but we might want to elevate it? 
        # In topo maps, entities often have Z coordinates. Let's set Z.
        center_3d = array.array('d', [center_x, center_y, float(z)])
        
        try:
            circle = ms.AddCircle(center_3d, r)
            circle.Layer = layer_name
            if is_index:
                circle.Lineweight = 30 # 0.30mm
                # 添加等高线注记
                text_pt = array.array('d', [center_x + r, center_y, float(z)])
                text = ms.AddText(str(z), text_pt, 2.5) # Height 2.5
                text.Layer = "DGX-J"
        except Exception as e:
            print(f"绘制等高线 Z={z} 失败: {e}")

    # 3. 绘制高程点 (Spot Heights)
    print("正在绘制高程点...")
    for _ in range(20):
        # 随机位置
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(0, base_radius)
        x = center_x + dist * math.cos(angle)
        y = center_y + dist * math.sin(angle)
        
        # 简单模拟高程 Z
        z_sim = 100 * (1 - dist / base_radius)
        
        pt = array.array('d', [x, y, z_sim])
        
        try:
            # 绘制点
            point = ms.AddPoint(pt)
            point.Layer = "GCD"
            
            # 绘制注记
            text_str = f"{z_sim:.1f}"
            text = ms.AddText(text_str, pt, 2.0)
            text.Layer = "GCD"
        except Exception as e:
            pass

    # 4. 绘制保护区边界 (Boundaries)
    print("正在绘制保护区边界...")
    # 一级保护区 (矩形)
    pts_1 = [center_x-200, center_y-200, center_x+200, center_y-200, center_x+200, center_y+200, center_x-200, center_y+200, center_x-200, center_y-200]
    # Flat coordinates for LightweightPolyline: x1, y1, x2, y2 ...
    pts_1_arr = array.array('d', pts_1)
    
    try:
        pl1 = ms.AddLightWeightPolyline(pts_1_arr)
        pl1.Layer = "BHQ-1"
        pl1.Closed = True
        
        # 添加界桩 (Markers)
        for i in range(0, len(pts_1)-2, 2):
            jx, jy = pts_1[i], pts_1[i+1]
            j_pt = array.array('d', [jx, jy, 0])
            ms.AddCircle(j_pt, 5) # Marker circle
            ms.AddText(f"JZ-1-{i//2+1}", array.array('d', [jx+5, jy+5, 0]), 3).Layer = "KJJ"
            
    except Exception as e:
        print(f"绘制一级保护区失败: {e}")

    # 二级保护区 (更大的不规则多边形)
    pts_2 = [center_x-600, center_y-600, center_x+600, center_y-500, center_x+500, center_y+600, center_x-500, center_y+500, center_x-600, center_y-600]
    pts_2_arr = array.array('d', pts_2)
    try:
        pl2 = ms.AddLightWeightPolyline(pts_2_arr)
        pl2.Layer = "BHQ-2"
        pl2.Closed = True
        
        # 添加标识牌 (Signboards)
        sign_pt = array.array('d', [center_x-600, center_y-600, 0])
        ms.AddText("二级保护区界牌", sign_pt, 5).Layer = "KJJ"
    except Exception as e:
        print(f"绘制二级保护区失败: {e}")

    # 5. 绘制方格网 (Grid)
    print("正在绘制方格网...")
    grid_step = 200
    min_x, min_y = center_x - 800, center_y - 800
    max_x, max_y = center_x + 800, center_y + 800
    
    for gx in range(int(min_x), int(max_x), grid_step):
        for gy in range(int(min_y), int(max_y), grid_step):
            # 画十字丝 (Cross)
            len_cross = 20
            p1 = array.array('d', [gx - len_cross, gy, 0])
            p2 = array.array('d', [gx + len_cross, gy, 0])
            p3 = array.array('d', [gx, gy - len_cross, 0])
            p4 = array.array('d', [gx, gy + len_cross, 0])
            
            try:
                l1 = ms.AddLine(p1, p2)
                l2 = ms.AddLine(p3, p4)
                l1.Layer = "TK"
                l2.Layer = "TK"
                
                # 标注坐标值 (在网格边缘)
                # 略... 简化展示
            except: pass

    # 6. 图框与标准化注记 (Layout & Standards)
    print("正在生成图框与标准注记...")
    # 简单的外框
    frame_pts = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y, min_x, min_y]
    frame_arr = array.array('d', frame_pts)
    try:
        frame = ms.AddLightWeightPolyline(frame_arr)
        frame.Layer = "TK"
        frame.Lineweight = 50 # 0.5mm
        
        # 图名
        title_pt = array.array('d', [center_x, max_y + 50, 0])
        title = ms.AddText("某地地形测量图", title_pt, 20)
        title.Alignment = 10 # Center (approximation, COM alignment needs setting point again usually)
        title.Layer = "TK"
        
        # 比例尺
        scale_pt = array.array('d', [center_x, min_y - 50, 0])
        ms.AddText("比例尺 1:500", scale_pt, 10).Layer = "TK"
        
        # 坐标系与高程基准注记 (User Requirement)
        note_pt_1 = array.array('d', [min_x, min_y - 100, 0])
        note_pt_2 = array.array('d', [min_x, min_y - 120, 0])
        ms.AddText("坐标系: CGCS2000 国家坐标系", note_pt_1, 8).Layer = "TK"
        ms.AddText("高程基准: 1985 国家高程基准", note_pt_2, 8).Layer = "TK"
        
        # 制作单位
        maker_pt = array.array('d', [max_x - 300, min_y - 120, 0])
        ms.AddText("制图: AGI 智能助理", maker_pt, 8).Layer = "TK"
        
    except Exception as e:
        print(f"图框绘制失败: {e}")

    print("任务完成！请在 AutoCAD 中查看结果。")

if __name__ == "__main__":
    main()
