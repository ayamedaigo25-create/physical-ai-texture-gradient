from pathlib import Path
import xml.etree.ElementTree as ET
import copy

home = Path.home()
root_dir = home / "pai_ws/texture_experiment"
textures = root_dir / "textures/train/episode_000005"
out_dir = root_dir / "worlds"
out_dir.mkdir(parents=True, exist_ok=True)

fuel = home / ".ignition/fuel/fuel.gazebosim.org/openrobotics/models"
table_src = fuel / "table/4/model.sdf"
cube_src = fuel / "wood%20cube%205cm/3/model.sdf"
world_src = home / "pai_ws/eye_in_hand/table_with_camera.sdf"

def get_model(path):
    model = ET.parse(path).getroot().find("model")
    assert model is not None, path
    return model

def set_texture(visual, image_path):
    old = visual.find("material")
    if old is not None:
        visual.remove(old)

    material = ET.SubElement(visual, "material")
    ET.SubElement(material, "ambient").text = "1 1 1 1"
    ET.SubElement(material, "diffuse").text = "1 1 1 1"

    pbr = ET.SubElement(material, "pbr")
    metal = ET.SubElement(pbr, "metal")
    ET.SubElement(metal, "albedo_map").text = str(image_path)
    ET.SubElement(metal, "metalness").text = "0"
    ET.SubElement(metal, "roughness").text = "0.8"

def physical_signature(model):
    return [
        ET.tostring(element, encoding="unicode")
        for element in model.iter()
        if element.tag in ("collision", "inertial", "static")
    ]

for condition in ("none", "sparse", "medium", "dense"):
    tree = ET.parse(world_src)
    world = tree.getroot().find("world")
    assert world is not None

    # 元のテーブル・キューブのincludeを除去
    removed = 0
    for inc in list(world.findall("include")):
        uri = (inc.findtext("uri") or "").lower()
        if "/models/table" in uri or "wood%20cube%205cm" in uri:
            world.remove(inc)
            removed += 1
    assert removed == 2, f"Expected two model includes: {removed}"

    # 元モデルの物理設定を維持して複製
    original_table = get_model(table_src)
    original_cube = get_model(cube_src)
    table = copy.deepcopy(original_table)
    cube = copy.deepcopy(original_cube)

    table.set("name", "experiment_table")
    cube.set("name", "experiment_cube")

    # 元ワールドの配置を維持
    for model, pose in (
        (table, "0.29 0 0 0 0 1.5708"),
        (cube, "0 -0.1 1.0 0 0 0"),
    ):
        old_pose = model.find("pose")
        if old_pose is not None:
            model.remove(old_pose)
        ET.SubElement(model, "pose").text = pose

    # 天板だけ変更。脚部は維持
    table_visual = table.find(".//visual[@name='visual1']")
    cube_visual = cube.find(".//visual[@name='visual']")
    assert table_visual is not None
    assert cube_visual is not None

    set_texture(table_visual, textures / f"floor_{condition}.png")
    set_texture(cube_visual, textures / f"object_{condition}.png")

    # Collision・Inertial・Staticを変更していないことを検証
    assert physical_signature(table) == physical_signature(original_table)
    assert physical_signature(cube) == physical_signature(original_cube)

    world.append(table)
    world.append(cube)

    destination = out_dir / f"{condition}.sdf"
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="unicode", xml_declaration=True)

    print(f"{condition}: {destination} [physical settings unchanged]")

print("All four worlds generated.")
