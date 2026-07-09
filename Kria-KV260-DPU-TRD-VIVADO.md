#A. Prerequisite
i. Install Vitis/VIVADO 2022.2 and Petalinux 2022.2

ii. Download Vitis AI(3.0) DPU TRD files: DPUCZDX8G_VAI_v3.0.tar.gz (link:https://www.xilinx.com/bin/public/openDownload?filename=DPUCZDX8G_VAI_v3.0.tar.gz)

iii. Download Petalinux 2022.2 BSP for KR260 (link:https://drive.google.com/drive/u/0/folders/13qcbGNd81VAhOV7NYlbpO_9hlCN2tkJL)

iv. Linux/Ubuntu PC with 20.04 LTS (preferred also for VIVADO/Vitis and Petalinux 2022.2 tools).

#B. VIVADO IP Design
rename the DPUCZDX8G_VAI_v3.0.tar.gz to DPU_TRD and place it inside Vitis-AI
tar -xzvf DPUCZDX8G_VAI_v3.0.tar.gz
mv -r ~/Downloads/DPUCZDX8G_VAI_v3.0 <Vitis AI path>/DPU_TRD
export TRD_HOME=<Vitis AI path>/DPU_TRD
cd $TRD_HOME
source <Vivado install path>/Vivado/2022.2/settings64.sh

##a.Customizing ZCU102 DPU TRD
nano ./prj/Vivado/hw/scripts/trd_prj.tcl
Edit following information in it:
dict set dict_prj dict_sys prj_name                 {KR260}
dict set dict_prj dict_sys prj_part                 {xck26-sfvc784-2LV-c}
dict set dict_prj dict_sys prj_board                {KR260}
dict set dict_prj dict_param DPU_CLK_MHz            {275}
dict set dict_prj dict_param DPU_NUM                {1}
dict set dict_prj dict_param DPU_ARCH               {512}
dict set dict_prj dict_param DPU_SFM_NUM            {0}
dict set dict_prj dict_param DPU_URAM_PER_DPU       {50}
^O enter ^X
In above tcl file, we are editing DPU ARCH for 512, it is for faster generation of the project. If your PC has 8+ Core of CPU and 16GB+ RAM then you can also do 4096(default).

Also edit : ./prj/Vivado/hw/scripts/base/trd_bd.tcl
nano ./prj/Vivado/hw/scripts/base/trd_bd.tcl
dict set dict_prj dict_param HP_CLK_MHz {274}
^O enter ^X

cd $TRD_HOME/prj/Vivado/hw
vivado -source scripts/trd_prj.tcl


