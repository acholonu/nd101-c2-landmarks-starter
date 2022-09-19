# Cloudformation

To successfully run this project, I need to use a machine with Graphic Processing Unit (GPU).  My current machine only has a CPU.  Thus I will use cloudformation to create a stack with a GPU-enabled EC2 instance, and remote connect (through ssh) to that instance to run my project.  Cloudformation is a way of writing Infrastructure As Code (IAC) so that you can create and bring down infrastructure easily and repeatibly. Since GPU enabled machines are so expensive, I wanted a quick way to do this and not have to remember all the settings.  It would become annoying if I had to do this configuration all the time through the console, especially if I wanted to make sure I did rack up costs on AWS.

## Stack Design

*INSERT LUCIDCHART IMAGE*

### Characteristics

- **EC2 Instance Type**: g4ad.4xlarge
- **Replication**: None.  I don't need fault tolerant because this is just used for running the associated notebook. It is not serving an application.  So when I am done, I will just throw everything away.
- **Subnets**: One large public subnet. I do not need to parcel out the ip's because I am running 1 specific job.
- **Security Group**: Ingress (incoming traffic) only from my IP through SSH.  Egress (outcoming traffic): All traffic.
- **AMI Image**:  I will pick an Amazon Linux Image already preloaded with python and other machine learning essentials. I may need need to create a requirements.txt file, and use ansible to install the image.
