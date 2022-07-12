import  os
from    os                  import  listdir
from    os                  import  walk
from    os.path             import  isfile, join
from    pathlib             import  Path

import  sys
import  glob
import  numpy as np
import  re
import  time
import  pudb
from    scipy               import  ndimage
# System dependency imports
import  nibabel              as      nib
import  pydicom              as      dicom
import  pylab
import  matplotlib.cm        as      cm

import  pfmisc
from    pfmisc._colors      import  Colors
from    pfmisc.message      import  Message
from scipy import ndimage
import pfmisc
import nibabel as nib
import os
import med2image
class med2image(object):
    """
        med2image accepts as input certain medical image formatted data
        and converts each (or specified) slice of this data to a graphical
        display format such as png or jpg.

    """

    _dictErr = {
        'inputFileFail'   : {
            'action'        :   'trying to read input file, ',
            'error'         :   'could not access/read file -- does it exist? Do you have permission?',
            'exitCode'      :   10},
        'emailFail'   : {
            'action'        :   'attempting to send notification email, ',
            'error'         :   'sending failed. Perhaps host is not email configured?',
            'exitCode'      :   20},
        'dcmInsertionFail': {
            'action'        :   'attempting insert DICOM into volume structure, ',
            'error'         :   'a dimension mismatch occurred. This DICOM file is of different image size to the rest.',
            'exitCode'      :   30},
        'ProtocolNameTag': {
            'action'        :   'attempting to parse DICOM header, ',
            'error'         :   'the DICOM file does not seem to contain a ProtocolName tag.',
            'exitCode'      :   40},
        'PatientNameTag':   {
            'action':           'attempting to parse DICOM header, ',
            'error':            'the DICOM file does not seem to contain a PatientName tag.',
            'exitCode':         41},
        'PatientAgeTag': {
            'action': '         attempting to parse DICOM header, ',
            'error':            'the DICOM file does not seem to contain a PatientAge tag.',
            'exitCode':         42},
        'PatientNameSex': {
            'action':           'attempting to parse DICOM header, ',
            'error':            'the DICOM file does not seem to contain a PatientSex tag.',
            'exitCode':         43},
        'PatientIDTag': {
            'action':           'attempting to parse DICOM header, ',
            'error':            'the DICOM file does not seem to contain a PatientID tag.',
            'exitCode':         44},
        'SeriesDescriptionTag': {
            'action':           'attempting to parse DICOM header, ',
            'error':            'the DICOM file does not seem to contain a SeriesDescription tag.',
            'exitCode':         45},
        'PatientSexTag': {
            'action':           'attempting to parse DICOM header, ',
            'error':            'the DICOM file does not seem to contain a PatientSex tag.',
            'exitCode':         46}
    }

    @staticmethod
    def mkdir(newdir, mode=0x775):
        """
        works the way a good mkdir should :)
            - already exists, silently complete
            - regular file in the way, raise an exception
            - parent directory(ies) does not exist, make them as well
        """
        if os.path.isdir(newdir):
            pass
        elif os.path.isfile(newdir):
            raise OSError("a file with the same name as the desired " \
                        "dir, '%s', already exists." % newdir)
        else:
            os.path(newdir).mkdir(parents = True, exist_ok = True)
        #     head, tail = os.path.split(newdir)
        #     if head and not os.path.isdir(head):
        #         os.mkdirs(head, exist_ok=True)
        #     if tail:
        #         os.mkdirs(newdir, exist_ok=True)

    def log(self, *args):
        '''
        get/set the internal pipeline log message object.

        Caller can further manipulate the log object with object-specific
        calls.
        '''
        if len(args):
            self._log = args[0]
        else:
            return self._log

    def name(self, *args):
        '''
        get/set the descriptive name text of this object.
        '''
        if len(args):
            self.__name = args[0]
        else:
            return self.__name

    def description(self, *args):
        '''
        Get / set internal object description.
        '''
        if len(args):
            self.str_desc = args[0]
        else:
            return self.str_desc

    @staticmethod
    def urlify(astr, astr_join = '_'):
        # Remove all non-word characters (everything except numbers and letters)
        # pudb.set_trace()
        astr = re.sub(r"[^\w\s]", '', astr)

        # Replace all runs of whitespace with an underscore
        astr = re.sub(r"\s+", astr_join, astr)

        return astr

    def __init__(self, **kwargs):

        #
        # Object desc block
        #
        self.str_desc                   = ''
        # self._log                        = msg.Message()
        # self._log._b_syslog              = True
        self.__name__                    = "med2image"

        # Directory and filenames
        self.str_workingDir             = ''
        self.str_inputFile              = ''
        self.lstr_inputFile             = []
        self.str_inputFileSubStr        = ''
        self.str_outputFileStem         = ''
        self.str_outputFileType         = ''
        self.str_outputDir              = ''
        self.str_inputDir               = ''

        self._b_convertAllSlices        = False
        self.str_sliceToConvert         = ''
        self.str_frameToConvert         = ''
        self._sliceToConvert            = -1
        self._frameToConvert            = -1

        self.str_stdout                 = ""
        self.str_stderr                 = ""
        self._exitCode                  = 0

        # The actual data volume and slice
        # are numpy ndarrays
        self._b_4D                      = False
        self._b_3D                      = False
        self._b_DICOM                   = False
        self.convertOnlySingleDICOM     = False
        self.preserveDICOMinputName     = False
        self._Vnp_4DVol                 = None
        self._Vnp_3DVol                 = None
        self._Mnp_2Dslice               = None
        self._dcm                       = None
        self._dcmList                   = []

        self.verbosity                  = 1

        # Flags
        self._b_showSlices              = False
        self._b_convertMiddleSlice      = False
        self._b_convertMiddleFrame      = False
        self._b_reslice                 = False
        self.func                       = None  # transformation function
        self.rot                        = '110'
        self.rotAngle                   = 90

        for key, value in kwargs.items():
            if key == "inputFile":              self.str_inputFile          = value
            if key == "inputFileSubStr":        self.str_inputFileSubStr    = value
            if key == "inputDir":               self.str_inputDir           = value
            if key == "outputDir":              self.str_outputDir          = value
            if key == "outputFileStem":         self.str_outputFileStem     = value
            if key == "outputFileType":         self.str_outputFileType     = value
            if key == "sliceToConvert":         self.str_sliceToConvert     = value
            if key == "frameToConvert":         self.str_frameToConvert     = value
            if key == "convertOnlySingleDICOM": self.convertOnlySingleDICOM = value
            if key == "preserveDICOMinputName": self.preserveDICOMinputName = value
            if key == "showSlices":             self._b_showSlices          = value
            if key == 'reslice':                self._b_reslice             = value
            if key == "func":                   self.func                   = value
            if key == "verbosity":              self.verbosity              = int(value)
            if key == "rot":                    self.rot                    = value
            if key == "rotAngle":               self.rotAngle               = int(value)

        # A logger
        self.dp                         = pfmisc.debug(
                                            verbosity   = self.verbosity,
                                            within      = self.__name__
                                            )
        self.LOG                        = self.dp.qprint

        if self.str_frameToConvert.lower() == 'm':
            self._b_convertMiddleFrame = True
        elif len(self.str_frameToConvert):
            self._frameToConvert = int(self.str_frameToConvert)

        if self.str_sliceToConvert.lower() == 'm':
            self._b_convertMiddleSlice = True
        elif len(self.str_sliceToConvert):
            self._sliceToConvert = int(self.str_sliceToConvert)

        if len(self.str_inputDir):
            self.str_inputFile  = '%s/%s' % (self.str_inputDir, self.str_inputFile)
        if not len(self.str_inputDir):
            self.str_inputDir = os.path.dirname(self.str_inputFile)
        if not len(self.str_inputDir): self.str_inputDir = '.'
        str_fileName, str_fileExtension  = os.path.splitext(self.str_outputFileStem)
        if len(self.str_outputFileType):
            str_fileExtension            = '.%s' % self.str_outputFileType

        if len(str_fileExtension) and not len(self.str_outputFileType):
            self.str_outputFileType     = str_fileExtension

        if not len(self.str_outputFileType) and not len(str_fileExtension):
            self.str_outputFileType     = 'png'

    def tic(self):
        """
            Port of the MatLAB function of same name
        """
        global Gtic_start
        Gtic_start = time.time()

    def toc(self, *args, **kwargs):
        """
            Port of the MatLAB function of same name

            Behaviour is controllable to some extent by the keyword
            args:


        """
        global Gtic_start
        f_elapsedTime = time.time() - Gtic_start
        for key, value in kwargs.items():
            if key == 'sysprint':   return value % f_elapsedTime
            if key == 'default':    return "Elapsed time = %f seconds." % f_elapsedTime
        return f_elapsedTime

    def run(self):
        '''
        The main 'engine' of the class.
        '''

    def echo(self, *args):
        self._b_echoCmd         = True
        if len(args):
            self._b_echoCmd     = args[0]

    def echoStdOut(self, *args):
        self._b_echoStdOut      = True
        if len(args):
            self._b_echoStdOut  = args[0]

    def stdout(self):
        return self.str_stdout

    def stderr(self):
        return self.str_stderr

    def exitCode(self):
        return self._exitCode

    def echoStdErr(self, *args):
        self._b_echoStdErr      = True
        if len(args):
            self._b_echoStdErr  = args[0]

    def dontRun(self, *args):
        self._b_runCmd          = False
        if len(args):
            self._b_runCmd      = args[0]

    def workingDir(self, *args):
        if len(args):
            self.str_workingDir = args[0]
        else:
            return self.str_workingDir

    def get_output_file_name(self, **kwargs):
        index   = 0
        frame   = 0
        str_subDir  = ""
        for key,val in kwargs.items():
            if key == 'index':  index       = val
            if key == 'frame':  frame       = val
            if key == 'subDir': str_subDir  = val

        if self._b_4D:
            str_outputFile = '%s/%s/%s-frame%03d-slice%03d.%s' % (
                                                    self.str_outputDir,
                                                    str_subDir,
                                                    self.str_outputFileStem,
                                                    frame, index,
                                                    self.str_outputFileType)
        else:
            if self.preserveDICOMinputName and (str_subDir == 'z' or str_subDir == ''):
                str_filePart    = os.path.splitext(self.lstr_inputFile[index])[0]
            else:
                str_filePart    = '%s-slice%03d' % (self.str_outputFileStem, index)
            str_outputFile      = '%s/%s/%s.%s' % (
                                        self.str_outputDir,
                                        str_subDir,
                                        str_filePart,
                                        self.str_outputFileType)
        return str_outputFile

    def dim_save(self, **kwargs):
        
		img_affine =data.affine

		print(sf)
		print('The img shape', img_np.shape[2])
		for i in range(img_np.shape[2]):
			slice_img_np = img_np[:,:,i]
			nft_img = nib.Nifti1Image(slice_img_np, img_affine)
			nib.save(nft_img, slice_dir_path + 'FLAIR_' + str(i) + '.nii.gz')

			if os.path.basename(sf) == '0':
				slice_img = nib.load(slice_dir_path + 'FLAIR_' + str(i) + '.nii.gz').get_data() / 5
				print('DID I GET HERE?')
				print('Writing to', str(i) + '.jpg') 
        dims            = data.shape
        str_dim         = 'z'
        b_makeSubDir    = False
        b_rot90         = False
        indexStart      = -1
        indexStop       = -1
        frame           = 0
        for key, val in kwargs.items():
            if key == 'dimension':  str_dim         = val
            if key == 'makeSubDir': b_makeSubDir    = val
            if key == 'indexStart': indexStart      = val
            if key == 'indexStop':  indexStop       = val
            if key == 'rot90':      b_rot90         = val
            if key == 'frame':      frame           = val

        str_subDir  = ''
        if b_makeSubDir:
            str_subDir = str_dim
            os.mkdir('c:\inetpub\wwwroot\%s' % (self.str_outputDir, str_subDir))

        dim_ix = {'x':0, 'y':1, 'z':2}
        if indexStart == 0 and indexStop == -1:
            indexStop = dims[dim_ix[str_dim]]
        self.LOG('Saving along "%s" dimension with %i degree rotation...' % (str_dim, self.rotAngle*b_rot90))
        for i in range(indexStart, indexStop):
            if str_dim == 'x':
                self._Mnp_2Dslice = data[i, :, :]
            elif str_dim == 'y':
                self._Mnp_2Dslice = data[:, i, :]
            else:
                self._Mnp_2Dslice = data[:, :, i]
            self.process_slice(b_rot90)
            
            str_outputFile="c:\\inetpub\\wwwroot\\"+self.get_output_file_name(index=i, subDir=str_subDir, frame=frame)
            if str_outputFile.endswith('dcm'):
                self._dcm = self._dcmList[i]
            self.slice_save("c://inetpub//wwwroot"+str_outputFile)
        self.LOG('%d images saved along "%s" dimension' % ((i+1), str_dim),
                end = '')
        if self.func:
            self.LOG(" with '%s' function applied." % self.func,
                syslog = False)
        else:
            self.LOG(".", syslog = False)

    def process_slice(self, b_rot90 = False):
        '''
        Processes a single slice.
        '''
        if b_rot90:
            self._Mnp_2Dslice = ndimage.rotate(self._Mnp_2Dslice, self.rotAngle)
        if self.func == 'invertIntensities':
            self.invert_slice_intensities()

    def slice_save(self, astr_outputFile):
        '''
        ARGS

        o astr_output
        The output filename.
        '''
        self.LOG('Input file = %s' % self.str_inputFile, level = 3)
        self.LOG('Outputfile = %s' % astr_outputFile, level = 3)
        fformat = astr_outputFile.split('.')[-1]
        if fformat == 'dcm':
            if self._dcm:
                self._dcm.pixel_array.flat = self._Mnp_2Dslice.flat
                self._dcm.PixelData = self._dcm.pixel_array.tostring()
                self._dcm.save_as(astr_outputFile)
            else:
                raise ValueError('dcm output format only available for DICOM files')
     #   else:
          #  pylab.imsave(astr_outputFile, self._Mnp_2Dslice, format=fformat, cmap = cm.Greys_r)

#    def invert_slice_intensities(self):
  #      '''
  #      Inverts intensities of a single slice.
   #     '''
 
 #       self._Mnp_2Dslice = self._Mnp_2Dslice*(-1) + self._Mnp_2Dslice.max()

class med2image_nii():
    '''
    Sub class that handles NIfTI data.
    '''

   

    def __init__(self, **kwargs):
        med2image.__init__(self, **kwargs)
        nimg = nib.load(r"C:\Users\Administrator\Desktop\nii\sub-A00000368_ses-20110101_task-gate_run-01_bold.nii.gz")
        data = nimg.get_data()
        if data.ndim == 4:
            self._Vnp_4DVol     = data
            self._b_4D          = True
        if data.ndim == 3:
            self._Vnp_3DVol     = data
            self._b_3D          = True

    def run(self):
        '''
        Runs the NIfTI conversion based on internal state.
        '''

        self.LOG('About to perform NifTI to %s conversion...\n' %
                  self.str_outputFileType)

        frames     = 1
        frameStart = 0
        frameEnd   = 0

        sliceStart = 0
        sliceEnd   = 0

        if self._b_4D:
            self.LOG('4D volume detected.\n')
            frames = self._Vnp_4DVol.shape[3]
        if self._b_3D:
            self.LOG('3D volume detected.\n')

        if self._b_convertMiddleFrame:
            self._frameToConvert = int(frames/2)

        if self._frameToConvert == -1:
            frameEnd    = frames
        else:
            frameStart  = self._frameToConvert
            frameEnd    = self._frameToConvert + 1

        for f in range(frameStart, frameEnd):
            if self._b_4D:
                self._Vnp_3DVol = self._Vnp_4DVol[:,:,:,f]
            slices     = self._Vnp_3DVol.shape[2]
            if self._b_convertMiddleSlice:
                self._sliceToConvert = int(slices/2)

            if self._sliceToConvert == -1:
                sliceEnd    = -1
            else:
                sliceStart  = self._sliceToConvert
                sliceEnd    = self._sliceToConvert + 1

            med2image.mkdir(self.str_outputDir)
            if self._b_reslice:
                for dim in ['x', 'y', 'z']:
                    self.dim_save(dimension = dim, makeSubDir = True, indexStart = sliceStart, indexStop = sliceEnd, rot90 = True, frame = f)
            else:
                self.dim_save(dimension = 'z', makeSubDir = False, indexStart = sliceStart, indexStop = sliceEnd, rot90 = True, frame = f)


'''
Sub class that handles NIfTI data.
'''
path="C:\\Users\\Administrator\\Desktop\\nii\\"
for i in os.listdir(path):
    nimg = nib.load(path+i)
    data = nimg.get_data()
    '''
    Runs the NIfTI conversion based on internal state.
    '''
    self=med2image()
    s=med2image_nii()

   # self.LOG('About to perform NifTI to %s conversion...\n' %self.str_outputFileType)

    frames     = 1
    frameStart = 0
    frameEnd   = 0

    sliceStart = 0
    sliceEnd   = 0

    if s._b_4D:
        s.LOG('4D volume detected.\n')
        frames = s._Vnp_4DVol.shape[3]
    if s._b_3D:
        s.LOG('3D volume detected.\n')

    if s._b_convertMiddleFrame:
        s._frameToConvert = int(frames/2)

    if s._frameToConvert == -1:
        frameEnd    = frames
    else:
        frameStart  = s._frameToConvert
        frameEnd    = s._frameToConvert + 1

    for f in range(frameStart, frameEnd):
        if s._b_4D:
            s._Vnp_3DVol = s._Vnp_4DVol[:,:,:,f]
    slices     = s._Vnp_3DVol.shape[2]
    if s._b_convertMiddleSlice:
        s._sliceToConvert = int(slices/2)

    if s._sliceToConvert == -1:
        sliceEnd    = -1
    else:
        sliceStart  = s._sliceToConvert
        sliceEnd    = s._sliceToConvert + 1

#    os.mkdir("C:\\inetpub\\wwwroot\\11")
    if s._b_reslice:
        for dim in ['x', 'y', 'z']:
            self.dim_save(dimension = dim, makeSubDir = True, indexStart = sliceStart, indexStop = sliceEnd, rot90 = True, frame = f)
    else:
        
        self.dim_save(dimension = 'z', makeSubDir = False, indexStart = sliceStart, indexStop = sliceEnd, rot90 = True, frame = f)



print(data)
