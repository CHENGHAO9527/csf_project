From  setuptools import setup , find_packages
with open ('requirements.txt') as f:
	Required = f.read().splitline()

setup(
	Name = 'Forest_Fire_Prediction'
	Author = 'Jiajun Wu, Cheng Miao, Hao Cheng'
	Description = ’predict the occurence of burned area'
	License =  'MIT License'
	Packages = find_packages()
	Install_requires = required
	Version = '0.0'

)
