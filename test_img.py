from PIL import Image     
import os       
path = './images/'
import shutil

if not os.path.exists('./deleted'):
    os.makedirs('./deleted')

print(len(os.listdir(path)))
counter = 0
with open('delete.txt', 'w+') as f:
    for file in os.listdir(path):      
         extension = file.split('.')[-1]
         if extension == 'jpg' or extension == 'png':
               counter+=1
               fileLoc = path+file
               img = Image.open(fileLoc)
               if img.mode != 'RGB':
                   pass
                   #shutil.move('./images/'+file, './deleted/')
                   #print("./annotations/"+file[0:-3]+'xml')
                   #shutil.move("./annotations/"+file[0:-3]+'xml', './deleted/')
         else:
             print(file)
print(counter)
