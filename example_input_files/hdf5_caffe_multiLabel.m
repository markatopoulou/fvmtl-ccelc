function outputFile = hdf5_caffe_multiLabel( imagePathFile, datasetName, chunksz, numOfConcepts, labelsetTrainFile, cor_file, stratificationFile)

labelMembership = dlmread(labelsetTrainFile);
outputFile = [datasetName '_' num2str(numOfConcepts) '_bs' num2str(chunksz) '.h5'];
correlation = true;
phi_cor = dlmread(cor_file);
disp('correlation file was loaded!');

disp('membership values will be used as semantic descriptors');
semanticDescriptor = dlmread(labelsetTrainFile);
T = numOfConcepts;

data_format = '/dataDF';
label_format = '/labelDF';

%shuffle the data
%Read the image path file
imagePathList = fopen(imagePathFile);
CimagePath = textscan(imagePathList, '%s %d');
fclose(imagePathList);
groundTruthRealPerImage=CimagePath{2};
imageRealPaths = CimagePath{1};

%comment the next three lines if you want to extract the shuffle image path list 
if exist(stratificationFile,'file')
    outputFile = [outputFile 'stratification'];

    randInd = dlmread(stratificationFile);
    randInd = randInd(:,end);
    imageRealPaths = imageRealPaths(randInd,:);
    groundTruthRealPerImage = groundTruthRealPerImage(randInd,:);
    semanticDescriptor = semanticDescriptor(randInd,:);
    labelMembership = labelMembership(randInd,:);
    % save the new image paths
%     fhTrain = fopen([outputFile '.txt'], 'wt');
    fhTrainKeyframes = fopen([outputFile '_keyframes.txt'], 'wt');
    for i = 1:size(imageRealPaths, 1)
        % save real image paths
         currentPath = [imageRealPaths{i} ' ' num2str(groundTruthRealPerImage(i)) double(sprintf('\n'));];
        fwrite(fhTrainKeyframes, currentPath, 'char*1');
    end 
    dlmwrite('semanticDescrTrain_stratification.txt', semanticDescriptor, ' ');
    dlmwrite('labelsetTrain_stratification.txt', labelMembership, ' ');
    fclose(fhTrainKeyframes);

else
    outputFile = [outputFile 'shuffle'];
    s=RandStream.create('mt19937ar','Seed',1); %Create a seed in order to take the same numbers
    RandStream.setGlobalStream(s);
    reset(s);
    randInd = randperm(size(imageRealPaths,1));
    imageRealPaths = imageRealPaths(randInd,:);
    groundTruthRealPerImage = groundTruthRealPerImage(randInd,:);
    semanticDescriptor = semanticDescriptor(randInd,:);
    labelMembership = labelMembership(randInd,:);
    % save the new image paths
%     fhTrain = fopen([outputFile '.txt'], 'wt');
    fhTrainKeyframes = fopen([outputFile '_keyframes.txt'], 'wt');
    for i = 1:size(imageRealPaths, 1)
        % save real image paths
         currentPath = [imageRealPaths{i} ' ' num2str(groundTruthRealPerImage(i)) double(sprintf('\n'));];
        fwrite(fhTrainKeyframes, currentPath, 'char*1');
    end 
    dlmwrite('semanticDescrTrain_shuffle.txt', semanticDescriptor, ' ');
    dlmwrite('labelsetTrain_shuffle.txt', labelMembership, ' ');
    fclose(fhTrainKeyframes);
end
% end shuffling

num_total_samples=size(groundTruthRealPerImage,1);
labelSize = size(phi_cor,2);

data_disk2d = zeros(T,num_total_samples);
% to simulate data being read from disk / generated etc.
label_disk=zeros(labelSize,num_total_samples); 

for i = 1:num_total_samples
    if mod(i,10000)==0
    disp(i);
    end
   
    data_disk2d(:,i) = semanticDescriptor(i,:);
    if nargin < 11
        aa = labelMembership(i,:);
        ind = find(aa==1);     
        phi_corTemp = phi_cor(ind,:);
        phi_corTemp = sum(phi_corTemp,1);
        if correlation
            phi_corTemp = phi_corTemp./size(ind,2);
        end
        label_disk(:,i)=phi_corTemp; 
%     else
%         label_disk(t,i)=1; 
    end
end

% which of the two is the correct format? % *data* is W*H*C*N 
data_disk = zeros(1,T,1,num_total_samples);
data_disk(1,:,1,:) = data_disk2d;

% data_disk = zeros(T,1,1,num_total_samples);
% data_disk(:,1,1,:) = data_disk2d;
data_disk2d = [];
created_flag=false;
totalct=0;
for batchno=1:num_total_samples/chunksz
  fprintf('batch no. %d\n', batchno);
  last_read=(batchno-1)*chunksz;

  % to simulate maximum data to be held in memory before dumping to hdf5 file 
  batchdata=data_disk(:,:,1,last_read+1:last_read+chunksz); 
  batchlabs=label_disk(:,last_read+1:last_read+chunksz);

  % store to hdf5
  startloc=struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
  curr_dat_sz=store2hdf5(outputFile, batchdata, batchlabs, ~created_flag, startloc, chunksz, data_format,label_format); 
  created_flag=true;% flag set so that file is created only once
  totalct=curr_dat_sz(end);% updated dataset size (#samples)
end

% added by me
remaining = mod(num_total_samples,chunksz);
% chunksz = remaining;
batchno = batchno+1;
if remaining>0
      fprintf('remaining: %d batch no. %d\n', chunksz, batchno);
  last_read=(batchno-1)*chunksz;

  % to simulate maximum data to be held in memory before dumping to hdf5 file 
  batchdata=data_disk(:,:,1,last_read+1:last_read+remaining); 
  batchlabs=label_disk(:,last_read+1:last_read+remaining);

  % store to hdf5
  startloc=struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
  curr_dat_sz=store2hdf5(outputFile, batchdata, batchlabs, ~created_flag, startloc, chunksz, data_format,label_format); 
  created_flag=true;% flag set so that file is created only once
  totalct=curr_dat_sz(end);% updated dataset size (#samples)
end
% end added by me

% display structure of the stored HDF5 file
h5disp(outputFile);

% also create a dataset suitable for the repvec caffe layer
%hdf5_caffe_MTL(num_total_samples, chunksz, [outputFile '_MTL']);
end