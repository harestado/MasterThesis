%-----------------------------------------------------------------------
% Job saved on 06-Nov-2023
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
% Creator: Oscar Alm Harestad
% Made to fit ERC-structured data
%-----------------------------------------------------------------------
%% Preparation

% The directory of the folder containing the subjects
data_dir = 'C:\Users\oscar\OneDrive - University of Bergen\Documents\Master\vsc\my_scan\fMRI_231207_ERC2VF_100\';

subjects = 1;

for s = subjects

    % Saves the images in easy-to-use variables for later
    anat = [data_dir '002_Sag_FSPGR_3D_192z_192x_TE_min_full\002_Sag_FSPGR_3D_192z_192x_TE_min_full.nii'];
    func = [data_dir '005_fMRI_default-pulsm_ling\005_fMRI_default-pulsm_ling.nii'];

    %% Preprocessing
    % The preprocessing steps themselves
    matlabbatch{1}.spm.spatial.realignunwarp.data.scans = {func};
    matlabbatch{1}.spm.spatial.realignunwarp.data.pmscan = '';
    matlabbatch{1}.spm.spatial.realignunwarp.eoptions.quality = 0.9;
    matlabbatch{1}.spm.spatial.realignunwarp.eoptions.sep = 4;
    matlabbatch{1}.spm.spatial.realignunwarp.eoptions.fwhm = 5;
    matlabbatch{1}.spm.spatial.realignunwarp.eoptions.rtm = 0;
    matlabbatch{1}.spm.spatial.realignunwarp.eoptions.einterp = 2;
    matlabbatch{1}.spm.spatial.realignunwarp.eoptions.ewrap = [0 0 0];
    matlabbatch{1}.spm.spatial.realignunwarp.eoptions.weight = '';
    matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.basfcn = [12 12];
    matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.regorder = 1;
    matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.lambda = 100000;
    matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.jm = 0;
    matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.fot = [4 5];
    matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.sot = [];
    matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.uwfwhm = 4;
    matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.rem = 1;
    matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.noi = 5;
    matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.expround = 'Average';
    matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.uwwhich = [2 1];
    matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.rinterp = 4;
    matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.mask = 1;
    matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.prefix = 'u';
    matlabbatch{2}.spm.spatial.coreg.estwrite.ref(1) = cfg_dep('Realign & Unwarp: Unwarped Mean Image', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','meanuwr'));
    matlabbatch{2}.spm.spatial.coreg.estwrite.source = {anat};
    matlabbatch{2}.spm.spatial.coreg.estwrite.other = {''};
    matlabbatch{2}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
    matlabbatch{2}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
    matlabbatch{2}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    matlabbatch{2}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
    matlabbatch{2}.spm.spatial.coreg.estwrite.roptions.interp = 4;
    matlabbatch{2}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
    matlabbatch{2}.spm.spatial.coreg.estwrite.roptions.mask = 0;
    matlabbatch{2}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';
    matlabbatch{3}.spm.spatial.preproc.channel.vols(1) = cfg_dep('Coregister: Estimate & Reslice: Coregistered Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','cfiles'));
    matlabbatch{3}.spm.spatial.preproc.channel.biasreg = 0.001;
    matlabbatch{3}.spm.spatial.preproc.channel.biasfwhm = 60;
    matlabbatch{3}.spm.spatial.preproc.channel.write = [0 1];
    matlabbatch{3}.spm.spatial.preproc.tissue(1).tpm = {'C:\Users\oscar\OneDrive - University of Bergen\Documents\Master\MATLAB\spm12\spm12\tpm\TPM.nii,1'};
    matlabbatch{3}.spm.spatial.preproc.tissue(1).ngaus = 1;
    matlabbatch{3}.spm.spatial.preproc.tissue(1).native = [1 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(1).warped = [0 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(2).tpm = {'C:\Users\oscar\OneDrive - University of Bergen\Documents\Master\MATLAB\spm12\spm12\tpm\TPM.nii,2'};
    matlabbatch{3}.spm.spatial.preproc.tissue(2).ngaus = 1;
    matlabbatch{3}.spm.spatial.preproc.tissue(2).native = [1 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(2).warped = [0 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(3).tpm = {'C:\Users\oscar\OneDrive - University of Bergen\Documents\Master\MATLAB\spm12\spm12\tpm\TPM.nii,3'};
    matlabbatch{3}.spm.spatial.preproc.tissue(3).ngaus = 2;
    matlabbatch{3}.spm.spatial.preproc.tissue(3).native = [1 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(3).warped = [0 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(4).tpm = {'C:\Users\oscar\OneDrive - University of Bergen\Documents\Master\MATLAB\spm12\spm12\tpm\TPM.nii,4'};
    matlabbatch{3}.spm.spatial.preproc.tissue(4).ngaus = 3;
    matlabbatch{3}.spm.spatial.preproc.tissue(4).native = [1 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(4).warped = [0 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(5).tpm = {'C:\Users\oscar\OneDrive - University of Bergen\Documents\Master\MATLAB\spm12\spm12\tpm\TPM.nii,5'};
    matlabbatch{3}.spm.spatial.preproc.tissue(5).ngaus = 4;
    matlabbatch{3}.spm.spatial.preproc.tissue(5).native = [1 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(5).warped = [0 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(6).tpm = {'C:\Users\oscar\OneDrive - University of Bergen\Documents\Master\MATLAB\spm12\spm12\tpm\TPM.nii,6'};
    matlabbatch{3}.spm.spatial.preproc.tissue(6).ngaus = 2;
    matlabbatch{3}.spm.spatial.preproc.tissue(6).native = [0 0];
    matlabbatch{3}.spm.spatial.preproc.tissue(6).warped = [0 0];
    matlabbatch{3}.spm.spatial.preproc.warp.mrf = 1;
    matlabbatch{3}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch{3}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{3}.spm.spatial.preproc.warp.affreg = 'mni';
    matlabbatch{3}.spm.spatial.preproc.warp.fwhm = 0;
    matlabbatch{3}.spm.spatial.preproc.warp.samp = 3;
    matlabbatch{3}.spm.spatial.preproc.warp.write = [0 1];
    matlabbatch{3}.spm.spatial.preproc.warp.vox = NaN;
    matlabbatch{3}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
                                                  NaN NaN NaN];
    matlabbatch{4}.spm.spatial.normalise.write.subj.def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
    matlabbatch{4}.spm.spatial.normalise.write.subj.resample(1) = cfg_dep('Realign & Unwarp: Unwarped Images (Sess 1)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{1}, '.','uwrfiles'));
    matlabbatch{4}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
                                                               78 76 85];
    matlabbatch{4}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
    matlabbatch{4}.spm.spatial.normalise.write.woptions.interp = 4;
    matlabbatch{4}.spm.spatial.normalise.write.woptions.prefix = 'no';
    
    

    % The "run"-function
    spm_jobman('run', matlabbatch);

end
