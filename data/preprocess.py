import json
import os
import shutil
from PIL import Image
import numpy as np
import matplotlib
from tqdm import tqdm

dir_imgs = "data/images/"

with open("data/SolarArrayPolygons.json") as f:
    polygon_dict = json.load(f)

# Parse all polygons.
parsed_polygons = {}
imgs_with_panels = set()
imgs_with_panels2 = set()

for polygon in polygon_dict["polygons"]:
    imgs_with_panels.add(polygon["image_name"])
    imgs_with_panels2.add(polygon["image_name"] + ".tif")
    # Check if the "image_name" key exists in the polygons variable.
    if polygon["image_name"] not in parsed_polygons:
        parsed_polygons[polygon["image_name"]] = {}
        parsed_polygons[polygon["image_name"]]["polygons"] = []
    parsed_polygons[polygon["image_name"]]["polygons"].append(
        polygon["polygon_vertices_pixels"]
    )

for imgs in os.listdir(dir_imgs):
    if (imgs) not in imgs_with_panels2:
        shutil.move(dir_imgs + imgs, "data/imgs_no_panel/")


def clip(num, size):
    num = int(num)
    if num > size:
        num = size
        print("MANUAL ALERT")
    if num < 0:
        num = 0
        print("MANUAL ALERT")
    return num


def tile_image_and_poly(img, parsed_polygons, size, name, output_dir, image_polygons):
    tiles = [
        img[x : x + size, y : y + size]
        for x in range(0, img.shape[0], size)
        for y in range(0, img.shape[1], size)
    ]

    i = 0
    j = 0
    points = []
    for i in range(200):
        for j in range(200):
            points.append([i, j])

    i = 0
    j = 0
    for tile in tiles:
        Image.fromarray(tile).save(output_dir + name + f"_{i}_{j}.tif")
        image_polygons[name + f"_{i}_{j}.tif"] = {}
        image_polygons[name + f"_{i}_{j}.tif"]["solar_panel"] = False
        image_polygons[name + f"_{i}_{j}.tif"]["solar_panel_count"] = 0
        image_polygons[name + f"_{i}_{j}.tif"]["polygons"] = []
        image_polygons[name + f"_{i}_{j}.tif"]["bounding_boxes"] = []
        for polygon in parsed_polygons[name]["polygons"]:
            # print("New Polygon")
            new_list = []
            hit = False

            if type(polygon[0]) == list:
                for number in polygon:
                    new_x, new_y = number[0] - size * (i), number[1] - size * (j)

                    if (new_x <= size and new_x >= 0) and (
                        new_y <= size and new_y >= 0
                    ):
                        hit = True

            else:
                new_x, new_y = polygon[0] - size * (i), polygon[1] - size * (j)
                if (new_x <= size and new_x >= 0) and (new_y <= size and new_y >= 0):
                    hit = True

            if hit:
                if type(polygon[0]) == list:
                    for number in polygon:
                        new_x, new_y = number[0] - size * (i), number[1] - size * (j)
                        new_list.append([round(new_x), round(new_y)])
                else:
                    new_x, new_y = polygon[0] - size * (i), polygon[1] - size * (j)
                    new_list.append([round(new_x), round(new_y)])

                path = matplotlib.path.Path(new_list)
                inside = path.clip_to_bbox([(0, 0), (size - 1, size - 1)])
                new_poly = inside.vertices
                area = inside.contains_points(points)
                area = area.sum()
                if area >= 25:
                    # print(i, j)
                    # print(area)

                    if image_polygons[name + f"_{i}_{j}.tif"]["solar_panel"] is False:
                        image_polygons[name + f"_{i}_{j}.tif"]["solar_panel"] = True

                    image_polygons[name + f"_{i}_{j}.tif"]["polygons"].append(
                        new_poly.tolist()
                    )
                    minX = min(new_poly, key=lambda x: x[0])[0]
                    maxX = max(new_poly, key=lambda x: x[0])[0]
                    minY = min(new_poly, key=lambda x: x[1])[1]
                    maxY = max(new_poly, key=lambda x: x[1])[1]

                    image_polygons[name + f"_{i}_{j}.tif"]["bounding_boxes"].append(
                        (minX, minY, maxX, maxY)
                    )
                    image_polygons[name + f"_{i}_{j}.tif"]["solar_panel_count"] += 1

                    # new_img = Image.fromarray(tile)
                    # draw = ImageDraw.Draw(new_img)
                    # new_poly = [tuple(x) for x in new_poly]
                    # draw.polygon(new_poly, outline="red")
                    # new_img.save(f"data/cool/cool_{i}_{j}_poly.tif")

                    # new_img = Image.fromarray(tile)
                    # draw = ImageDraw.Draw(new_img)
                    # draw.rectangle(((minX, minY), (maxX, maxY)), outline="red")
                    # new_img.save(f"data/cool/cool_{i}_{j}_bbox.tif")

                # print(new_poly)

        i += 1
        if i % (img.shape[0] / size) == 0:
            i = 0
            j += 1
    return image_polygons


image_polygons = {}
for imgs in tqdm(os.listdir(dir_imgs)):
    im = Image.open(dir_imgs + imgs)
    im = np.asarray(im)
    image_polygons = tile_image_and_poly(
        im, parsed_polygons, 200, imgs[:-4], "data/processed_imgs/", image_polygons
    )

with open("data/image_polygons.json", "w", encoding="utf-8") as f:
    json.dump(image_polygons, f, ensure_ascii=False, indent=4)
