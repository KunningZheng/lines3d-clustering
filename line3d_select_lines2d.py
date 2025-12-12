from preprocess.line3dpp_loader import parse_line_segments
from preprocess.sfm_reader import load_sparse_model
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

def save_segments_l3dpp(lines, output_folder, img_name, width, height):
    """
    将线段以 Boost-Serialization 风格保存为 XML 文件 (.bin)
    lines: (N, 4) [x1, y1, x2, y2]
    """
    import xml.dom.minidom as minidom

    filename = f'segments_L3D++_{img_name}_{width}x{height}.bin'
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, filename)

    num_lines = len(lines)

    # 创建 XML 结构
    root = ET.Element('boost_serialization', {
        'signature': 'serialization::archive',
        'version': '18'
    })
    data = ET.SubElement(root, 'data', {
        'class_id': '0',
        'tracking_level': '0',
        'version': '0'
    })

    ET.SubElement(data, 'width_').text = str(num_lines)
    ET.SubElement(data, 'height_').text = '1'
    ET.SubElement(data, 'real_width_').text = str(num_lines)
    ET.SubElement(data, 'pitchCPU_').text = str(num_lines * 16)
    ET.SubElement(data, 'strideCPU_').text = str(num_lines)
    ET.SubElement(data, 'pitchGPU_').text = '0'
    ET.SubElement(data, 'strideGPU_').text = '0'

    for i, line in enumerate(lines):
        if i == 0:
            item = ET.SubElement(data, 'item', {
                'class_id': '1',
                'tracking_level': '0',
                'version': '0'
            })
        else:
            item = ET.SubElement(data, 'item')
        ET.SubElement(item, 'x').text = f"{line[0]:.9e}"
        ET.SubElement(item, 'y').text = f"{line[1]:.9e}"
        ET.SubElement(item, 'z').text = f"{line[2]:.9e}"
        ET.SubElement(item, 'w').text = f"{line[3]:.9e}"

    # 使用 minidom 格式化（不写入声明）
    xml_str = ET.tostring(root, encoding='utf-8')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="\t", encoding='utf-8').decode('utf-8')

    # 去掉 minidom 自动添加的 XML 声明
    pretty_xml = "\n".join(line for line in pretty_xml.splitlines() if not line.strip().startswith("<?xml"))

    # 去掉最外层的一个制表符层级
    fixed_lines = []
    for line in pretty_xml.splitlines():
        if line.startswith("\t"):
            fixed_lines.append(line[1:])  # 删除一个制表符
        else:
            fixed_lines.append(line)
    pretty_xml = "\n".join(fixed_lines)

    # 写入文件（仅保留我们定义的头部）
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n')
        f.write('<!DOCTYPE boost_serialization>\n')
        f.write(pretty_xml)

    return filepath



line3dpp_folder = "/home/rylynn/Pictures/LinesDetection_Workspace/datasets/group611_800z/sparse_txt/Line3D++"
sparse_model_path = "/media/rylynn/data/datasets_2Dline/group611_800z_workspace/dense/sparse/"

camerasInfo, _, _ = load_sparse_model(sparse_model_path)
for cam_id, cam_dict in tqdm(enumerate(camerasInfo), desc="Processing lines"):
    width = int(cam_dict['width'])
    height = int(cam_dict['height'])
    segments = parse_line_segments(line3dpp_folder, cam_id+1, width, height)[:, [1,0,3,2]]

    # 取长度前3000的线段
    segments_lengths = ((segments[:, 0]-segments[:, 2])**2 + (segments[:, 1]-segments[:, 3])**2)**0.5
    topk_indices = segments_lengths.argsort()[-3000:][::-1]
    segments_topk = segments[topk_indices]

    output_folder = os.path.join(line3dpp_folder, 'L3D++_data_new')
    os.makedirs(output_folder, exist_ok=True)
    filepath = save_segments_l3dpp(segments_topk, output_folder, cam_id+1, width, height)
