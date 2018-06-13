function hdf5_fromCovarianceMatrix(corFile, datasetName)

outputFile = [datasetName '_covariance.h5'];
data_format = '/dataDFcov';
label_format = '/labelDFcov';
phi_cor = dlmread(corFile);
T = size(phi_cor,1) * size(phi_cor,1);
labelSize = size(phi_cor,1);
phi_cor = reshape(phi_cor,[1,T]);
% T = 60*60;
% phi_cor = rand(1,T);

data_disk2d = zeros(T,1);
label_disk=zeros(labelSize,1); 

data_disk2d(:,1) = phi_cor;
label_disk(:,1)=rand(1,labelSize); 

% which of the two is the correct format? % *data* is W*H*C*N 
data_disk = zeros(T,1,1,1);
data_disk(:,1,1,:) = data_disk2d;

created_flag=false;
totalct=0;

% store to hdf5
startloc=struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
curr_dat_sz=store2hdf5(outputFile, data_disk, label_disk, ~created_flag, [], [], data_format,label_format); 

% display structure of the stored HDF5 file
h5disp(outputFile);

% read the file
hinfo = hdf5info(outputFile);
dset = hdf5read(hinfo.GroupHierarchy.Datasets(1));