1.PhysioNet和Yaseen对应两个数据集的文件夹
2.打开code 文件夹中的test.ipynb文件。导入model中的model.py文件，然后使用model.load_state_dict()) 加载model文件夹里面的.pth权重。
3.加载数据集：使用np.load（）加载test_data文件夹中的npz文件，里面两个属性features为数据，labels为标签。
4.运行下面的测试代码进行测试



1.PhysioNet and Yaseen correspond to the folders of the two datasets
2. Open the test.ipynb file in the code folder. Import the model.py file in the model, and then use model.load_state_dict()) to load the.pth weights in the model folder.
3. Loading the dataset: Use np.load () to load the npz file in the test_data folder. There are two attributes inside, "features" for data and "labels" for labels.
4. Run the following test code for testing
