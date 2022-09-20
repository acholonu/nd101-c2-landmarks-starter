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

### Files

- **create_stack.sh**: Bash script to run to create AWS stack defined in ml_workspace.yml. Arguments to pass:
  - Stack Name: the name of the stack
  - CloudFormation file: name of the cloudformation file
  - Parameters File: the name of the parameter file
- **delete_stack.sh**: Bash script to delete a specific AWS stack.
- **ml_workspace.yml**: CloudFormation code that defines that networking and EC2 instance I need to handle the deep learning workflow. Infrastructure as code (IaC). Really just need a cheap GPU-enabled EC2 instance.

### Minor Steps

You must make sure create_stack.sh and delete_stack.sh are executables. To make these files executable, use the following commands:

```bash

chmod +x create_stack.sh
chmod +x delete_stack.sh
```
