Figures for the guide (development participants only; descriptive, not results).
mri_examples.png              the 16 axial slices the model uses, one CDR 0 and one CDR > 0 participant of similar age
mri_examples_middle.png       one larger slice (axial 89) for the same two people
preprocessing_before_after.png raw voxels -> brain-normalised [0,1] -> 224x224 model input, with intensity histograms
slice_budgets.png             which slices the 4 / 8 / 16 budgets use, with their levels on a sagittal view
features_cnn_heatmap.png      frozen ResNet-18 output: 16 slices x 512 numbers per person (z-scored per feature)
features_vit_heatmap.png      frozen ViT-B/16 output: 16 slices x 768 numbers per person (z-scored per feature)
features_pooled.png           mean over slices, CNN 512 + ViT 768 = the 1280-number fusion vector the classifier gets
features_pca.png              all development participants' fusion vectors in 2-D (PCA), coloured by CDR group
features_table.csv            shapes and summary numbers of the pooled vectors for the two example people
