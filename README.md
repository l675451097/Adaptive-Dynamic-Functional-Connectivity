# Adaptive-Dynamic-Functional-Connectivity
The code of “Deep Spatial-Temporal Feature Fusion from Adaptive Dynamic Functional Connectivity for MCI Classification” doi.org/10.1109/tmi.2020.2976825

fmri_net_built对应多模态融合，其输出net为后续代码中的low_net （即section III-A）

highorder_net_built 对应RLS自适应动态连接计算+高阶网络构建 （即section III-B + D的一部分）

temporal_feature_extraction 对应论文的Spatial Feature Extraction（即section III-C）

spatial_feature_extraction 对应论文的Temporal Feature Extraction （即section III-D的一部分）

其余data_for_ultra，ffols_gui.mexw64，FFRLS，net_built_ultar_lasso_OLS是主函数的子函数

MMD-AE文件中是MMD-AE深度融合模型（即section III-E的一部分）
