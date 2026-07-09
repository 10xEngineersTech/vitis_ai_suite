# A. Prerequisite
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

dict set dict_prj dict_sys prj_name                 {KV260}
dict set dict_prj dict_sys prj_part                 {xck26-sfvc784-2LV-c}
dict set dict_prj dict_sys prj_board                {KV260}
dict set dict_prj dict_param DPU_CLK_MHz            {275}
dict set dict_prj dict_param DPU_NUM                {1}
dict set dict_prj dict_param DPU_ARCH               {4096}
dict set dict_prj dict_param DPU_SFM_NUM            {0}
dict set dict_prj dict_param DPU_URAM_PER_DPU       {50}
^O enter ^X


Also edit : ./prj/Vivado/hw/scripts/base/trd_bd.tcl
nano ./prj/Vivado/hw/scripts/base/trd_bd.tcl
dict set dict_prj dict_param HP_CLK_MHz {274}
^O enter ^X

cd $TRD_HOME/prj/Vivado/hw
vivado -source scripts/trd_prj.tcl

it will open the vivado somthing like this
<img width="1542" height="869" alt="img1" src="https://github.com/user-attachments/assets/482dcf36-2efd-4a05-8d2a-16c98a734616" />

now there are a lot of things to corrct.
lets move 1 by 1.
first go to settings, the following window should be open
<img width="1808" height="981" alt="img2" src="https://github.com/user-attachments/assets/f6116666-210f-4f9a-9442-c1fae06bd95e" />

then click on 3 dots infron of project device
then search "kv260" -> click on onnections -> click on down arrow and select "AI stater Kit carrier card" as seen in the following image
<img width="1808" height="981" alt="img3" src="https://github.com/user-attachments/assets/36001403-2e06-4cd4-a9c2-50b124323dcf" />

now check the box againset Project is an extensiable Vitis Platform 
now yor settings gernale tab should look like this
<img width="1808" height="981" alt="img4" src="https://github.com/user-attachments/assets/51f396c4-571b-4d53-a41a-5438709de5d0" />

goto setting Bitstream tab, and select bin_file option like following
<img width="1808" height="981" alt="Screenshot from 2026-07-09 18-52-59" src="https://github.com/user-attachments/assets/f32f15d2-e3b0-48d4-bb5a-01ad3ed72143" />

apply and ok

now you will see a light-yellow bar at the top of you vivado window saying something like
"The design has 6 blocks that should be upgraded" you need to click on "6 blocks" that is clickable and in blue color. at the bottom console window area you will be able to see all these 6 blocks, you need to select all of these and click on update selected.
as seen in this image
<img width="1808" height="981" alt="img8" src="https://github.com/user-attachments/assets/abe7ee94-44a2-4875-a716-61ebca78b903" />

now you need to delete the wire that is selected in the following image
<img width="1808" height="981" alt="img6" src="https://github.com/user-attachments/assets/9b8f6db3-6c82-44e0-86f5-faee4f21e288" />

now remove all the extra ports of ZYNQ UltraScale Block and click on "Run:Connection Automation" and Ok like following
<img width="1808" height="981" alt="img6" src="https://github.com/user-attachments/assets/6317b470-2b57-47b7-b406-1ed485259f84" />

click on save and Regenrate layout, now your design will somehow look like following
<img width="1810" height="918" alt="image" src="https://github.com/user-attachments/assets/f22941fa-6660-4209-9515-aced6620a047" />

now just beside the Diagram file, click on Platform Setup, goto AXI Port and select a master Axi port in the zynq section like following
<img width="1808" height="981" alt="Screenshot from 2026-07-09 18-58-45" src="https://github.com/user-attachments/assets/4b472c8b-1efa-4185-b02c-bdbebc99e5f8" />

now navigate to AXI Port to Clock section, enable the pl_clk0 and make it default like following.
<img width="1808" height="981" alt="Screenshot from 2026-07-09 18-57-54" src="https://github.com/user-attachments/assets/63e0a1b0-c10a-4a28-8bc2-fcc1cbcb650c" />

now save and validate you project and click on create bitstream.

