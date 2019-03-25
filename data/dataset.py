import os
from PIL import Image
import torch as t
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import DataLoader

transform=T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
    #T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
])

class Eventbased(data.Dataset):
    def __init__(self,img_root,event_root,test=False,transforms=None):
        self.test=test
        self.transforms = transforms
        if test==False:
            imglist=os.listdir(img_root)
            eventlist=os.listdir(event_root)
            self.imgs=[os.path.join(img_root,img) for img in imglist]
            self.events=[os.path.join(event_root,eventpic) for eventpic in eventlist]
        else:
            eventlist = os.listdir(event_root)
            self.events = [os.path.join(event_root, eventpic) for eventpic in eventlist]
    def __getitem__(self,index):
        if self.test==False:
            img_path=self.imgs[index]
            graypic=Image.open(img_path)
            label=t.ones(1)
            event_path1 = self.events[index*3]
            event_path2 = self.events[index * 3+1]
            event_path3 = self.events[index * 3+2]
            eventpic1 = Image.open(event_path1)
            eventpic2 = Image.open(event_path2)
            eventpic3 = Image.open(event_path3)
            if self.transforms:
                graypic=self.transforms(graypic)
                eventpic1 =self.transforms(eventpic1)
                eventpic2 =self.transforms(eventpic2)
                eventpic3 =self.transforms(eventpic3)
            eventpic=t.cat([eventpic1,eventpic2,eventpic3],0)
            return graypic,eventpic,label
        else:
            event_path1 = self.events[index]
            event_path2 = self.events[index]
            event_path3 = self.events[index]
            eventpic1 = Image.open(event_path1)
            eventpic2 = Image.open(event_path2)
            eventpic3 = Image.open(event_path3)
            if self.transforms:
                eventpic1 = self.transforms(eventpic1)
                eventpic2 = self.transforms(eventpic2)
                eventpic3 = self.transforms(eventpic3)
            eventpic = t.cat([eventpic1, eventpic2, eventpic3], 0)
            name=str(event_path1.split('/')[-1])
            return eventpic,name
    def __len__(self):
        if self.test==False:
            return len(self.imgs)
        else:
            return len(self.events)

if __name__=='__main__':
    test_dataset=Eventbased('./urban/images/','./urban/recovery/',test=True,transforms=transform)

    testloader=DataLoader(test_dataset,batch_size=1,shuffle=False)
    eventpic,name=test_dataset[2]
    print(eventpic.size(),name)


