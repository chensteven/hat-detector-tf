from PIL import Image, ImageFile
import glob
images = glob.glob('./images/*')
print(len(images))
for image in images:
    new = image[0:-3]
    img = Image.open(image)
    img.save(new.replace('images', 'new_images') +'jpg', "JPEG", quality=80, optimize=True, progressive=True)
