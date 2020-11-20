import os
from pathlib import Path

import xmltodict

from core.comicspage import ComicsPage

ORIGIN_PATH = Path("assets/origin")
PAGES_PATH = Path("assets/pages")
GT_PATH = Path("assets/gt")

def convert_svg_to_json(filename: Path):
    res = xmltodict.parse(filename.read_text(encoding="utf-8"))

    root = res["svg"]
    title = root["title"]
    contents = root["svg"]

    width = contents[0]["image"]["@width"]
    height = contents[0]["image"]["@height"]
    href = contents[0]["image"]["@xlink:href"]

    base_name = os.path.basename(href)
    href = str(PAGES_PATH.joinpath(base_name).as_posix())

    # print(href)

    bubbles = {}
    # print(contents)

    for content in contents:
        if content["@class"] == "Balloon":
            # print(content)
            polygon = content["polygon"]

            for pol in polygon:
                # print(pol["@points"])
                bubbles[pol["@id"]] = {
                    "points": [list(map(lambda x: int(x), el.split(","))) for el in pol["@points"].split(" ")],
                    "metadata": {
                        "borderStyle": pol["metadata"]["@borderStyle"],
                        "tailTip": pol["metadata"]["@tailTip"],
                        "boundingBox": pol["metadata"]["@boundingBox"],
                        "tailDirection": pol["metadata"]["@tailDirection"],
                        "rank": pol["metadata"]["@rank"]
                    },
                    "lines": [],
                }
        elif content["@class"] == "Line":
            polygon = content["polygon"]
            for pol in polygon:
                if "@idBalloon" in pol["metadata"]:
                    key = pol["metadata"]["@idBalloon"]
                    # print(pol["@points"])
                    bubbles[key]["lines"].append({
                        "points": [list(map(lambda x: int(x), el.split(","))) for el in pol["@points"].split(" ")],
                        "textType": pol["metadata"]["@textType"],
                        "text": pol["metadata"]["#text"]
                    })

    speech_bubbles = []

    for key, value in bubbles.items():
        bubble = value
        bubble["id"] = key
        speech_bubbles.append(bubble)

    result = {
        "title": title,
        "width": width,
        "height": height,
        "href": href,
        "speechbubbles": speech_bubbles
    }

    destination = Path(ORIGIN_PATH, title + ".json")
    ComicsPage(result).save(destination)


for filename in GT_PATH.glob("*.svg"):
    try:
        convert_svg_to_json(filename)
    except (RuntimeError, TypeError, NameError, KeyError) as err:
        print(err)
        print(filename)


def get_all_configs():
    configs = []
    for filename in ORIGIN_PATH.glob("*.json"):
        configs.append(ComicsPage.open(filename))
    return configs
