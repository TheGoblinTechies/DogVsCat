# DogVsCat
The second project:
Well, this is the second project in Winter. It's about Dog vs Cat in Kaggle competition.
Considering limited computation power, Transfer Learning should be the best for this problem. Installing an OpenCv for python is also a quite tough mission, so I have already use dog.py (which has a reference from a zhihu user name Wu Ji, at https://zhuanlan.zhihu.com/p/27547647) to extract the feature from the dataset. I manually choose some images in train set as test set :D
So you could just run classifier.py to do the classifition mission.

What you should do for this project is to understand why we could do this, and achieve almost 100% of accuracy on the test set. And here are some interesting questions:
1. Why ResNet is so famours and powerful (which is my favorite paper :P)?
2. Why only 1000 images (actually it's a quite small dataset) could have such good performance? How do the "ImageNet" work in this project?
3. Question 2 is a way called Fune-tune. Can you list other methods to fune-tune this network, like froze some layers and fune-tune other layers?
4. What if there are fewer images in train set, like 10, or even none? This is quite famours in Unsupervised Learning, called Zero/Few Shot Learning.
5. For the code, there is a way called data-augmentation. You could apply this during extracting features, to make ten feel like hundreds.

Sorry for late releasing this code, bugs are quite a lot in this code, please to double check the code and report the bugs in group-chat, I really appreciate it for this :p
 
