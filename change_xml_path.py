import glob
import xml.etree.ElementTree as ET


def change_xml_path(xml):
    prefix = '~/hat-detector/dataset/images/'
    tree = ET.parse(xml)
    root = tree.getroot()
    for og_path in root.iter('path'):
        path = og_path.text
        print(path)
        new_base = path.split('/')[-1]
        new_path = prefix + new_base
        og_path.text = new_path
    tree.write(xml)

def main():
    xmls = glob.glob('./dataset/annotations/*.xml')
    for xml in xmls:
        change_xml_path(xml)
    print(len(xmls))

main()
