# FalCAN

- Here is the [C code](https://github.com/Ipsitakoley/FalCAN/blob/main/target_ICCPS.c) to find out the attackable IDs and attack window lengths of their instances
- The CAN traffic logs are for the benchmark traffic from 2011 Chevy-Impala [1](https://www.usenix.org/conference/usenixsecurity23/presentation/serag)
- We analyse the traffic for 500kbps and 250 kbps baud rates.
- We analyse the traffic for [high busload](https://github.com/Ipsitakoley/FalCAN/blob/main/log_can_500_hbl.csv) (i.e. in the presence of all the messages) and [low busload](https://github.com/Ipsitakoley/FalCAN/blob/main/log_can_500_lbl.csv)(in the presence of 50% tx from each ECU)
- The following video shows how FalCAN (black plots) attack sequence injected to gradually manipulate the acceleration data (in green) changes the speed (in red) for an adaptive cruise controller.
https://github.com/Ipsitakoley/FalCAN/blob/main/fdi_carmaker_FalCAN_500.mp4
# CS637-Embedded-Systems
