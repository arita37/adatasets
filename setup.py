# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""


"""
import io, os, subprocess, sys
from setuptools import find_packages, setup

######################################################################################
root = os.path.abspath(os.path.dirname(__file__))



##### Version  #######################################################################
reponame = "adatasets"
version ='0.0.1'
cmdclass= None
print("version", version)



##### Requirements ###################################################################
#with open('install/requirements.txt') as fp:
#    install_requires = fp.read()
install_requires = ['pyyaml', 'pmlb', 'utilmy']



###### Description ###################################################################
#with open("README.md", "r") as fh:
#    long_description = fh.read()

def get_current_githash():
   import subprocess 
   # label = subprocess.check_output(["git", "describe", "--always"]).strip();   
   label = subprocess.check_output([ 'git', 'rev-parse', 'HEAD' ]).strip();      
   label = label.decode('utf-8')
   return label

githash = get_current_githash()


long_description =  f"""

```
Utils

git hash : https://github.com/arita37/{reponame}/tree/{githash}


```
"""



### Packages  ########################################################
packages = [reponame] + [ reponame+ "." + p for p in find_packages(reponame)]
print(packages)


scripts = [     ]



### CLI Scripts  ###################################################   
entry_points={ 'console_scripts': [

              ] }






##################################################################   
setup(
    name=reponame,
    description="utils datasets",
    keywords='utils',
    
    author="Nono",    
    install_requires=install_requires,
    python_requires='>=3.6.5',
    
    packages=packages,

    include_package_data=True,
    #    package_data= {'': extra_files},

    package_data={
       '': ['*','*/*','*/*/*','*/*/*/*']
    },

   
    ### Versioning
    version=version,
    #cmdclass=cmdclass,


    #### CLI
    scripts = scripts,
  
    ### CLI pyton
    entry_points= entry_points,


    long_description=long_description,
    long_description_content_type="text/markdown",


    classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: ' +
          'Artificial Intelligence',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: ' +
          'Python Modules',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Environment :: Console',
          'Environment :: Web Environment',
          'Operating System :: POSIX',
          'Operating System :: MacOS :: MacOS X',
      ]
)








