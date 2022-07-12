from genericpath import exists
import nilearn.plotting as plot
import nilearn.image as I
import os,gzip,io
import nibabel as nib
path="C:/users/administrator/desktop/nii"
path2="C:/users/administrator/desktop/nii/out/"
for i in os.listdir(path):
    if(".nii.gz" in i):
        pass
    else:

        if(".nii" in i):
            img = nib.load(path+"/"+i)
            data = img.get_fdata()
            print(data)
            import imageio
            X=0
            for s in data:
                import numpy
                aleph=numpy.array(s,dtype=numpy.int8)
                X=X+1
                t=1000
                if(not exists("c:/inetpub/wwwroot/stat/"+"_"+i+str(X)+"/")):

                    os.mkdir("c:/inetpub/wwwroot/stat/"+"_"+i+str(X)+"/")
                if(not exists("c:/inetpub/wwwroot/out/"+"_"+i+str(X)+"/")):

                #    os.mkdir("c:/inetpub/wwwroot/stat/"+"_"+i+str(X)+"/")
                
                    os.mkdir("c:/inetpub/wwwroot/out/"+"_"+i+str(X)+"/")
#                os.mkdir("c:/inetpub/wwwroot/stat/")
                for h in I.iter_img(img):
                    t=t+1
    # img is now an in-memory 3D img
                    plot.plot_stat_map(h, threshold=3, display_mode="z", cut_coords=1,
                                        colorbar=True,output_file="c:/inetpub/wwwroot/stat/"+"_"+i+str(X)+"/"+str(t))

                    plot.plot_img(img=h,colorbar=True,output_file="c:/inetpub/wwwroot/out/"+"_"+i+str(X)+"/"+str(t))
               #     plot.plot_img(aleph)
                    


 