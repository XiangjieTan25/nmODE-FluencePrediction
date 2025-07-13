import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint


MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver

class ODEBlock(nn.Module):
    def __init__(self, odefunc, tol=1e-3, adjoint=False):
        """
        Code adapted from https://github.com/EmilienDupont/augmented-neural-odes

        Utility class that wraps odeint and odeint_adjoint.

        Args:
            odefunc (nn.Module): the module to be evaluated
            tol (float): tolerance for the ODE solver容差：容许误差
            adjoint (bool): whether to use the adjoint method for gradient calculation
        """
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None):
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe = 0

        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float()
        else:
            integration_time = eval_times.type_as(x)
            
        if self.adjoint:
            out = odeint_adjoint(self.odefunc, x, integration_time,
                                 rtol=self.tol, atol=self.tol, method='dopri5',
                                 options={'max_num_steps': MAX_NUM_STEPS})
        else:
            out = odeint(self.odefunc, x, integration_time,
                         rtol=self.tol, atol=self.tol, method='dopri5',
                         options={'max_num_steps': MAX_NUM_STEPS})

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out[1]

    def trajectory(self, x, timesteps):
        '''
        用于计算ODE的轨迹。它接收输入x和timesteps参数，根据timesteps生成积分时间，并调用forward方法进行求解。


        '''
        integration_time = torch.linspace(0., 1., timesteps)
        return self.forward(x, eval_times=integration_time)
    

class nmODE(nn.Module):
    def __init__(self):
        """
        """
        super(nmODE, self).__init__()
        self.nfe = 0  # Number of function evaluations
        self.gamma = None
    
    def fresh(self, gamma):
        self.gamma = gamma
    
    def forward(self, t, p):
        self.nfe += 1
        dpdt = -p + torch.pow(torch.sin(p + self.gamma), 2)
        return dpdt



class att_Unet1(nn.Module):
    def __init__(self,in_chnl=5,out_chnl=1,ifode = True, tol=1e-3, adjoint=False,eval_times = (0, 1)):
        super(att_Unet1,self).__init__()
        
        self.input = nn.Sequential(
            nn.Conv2d(in_chnl,128,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.sec_conv = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(256,512,kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.third_conv = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(512,1024,kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.four_conv = nn.Sequential(
            nn.Conv2d(1024,1024,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024,1024,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(1024,512,kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.third_conv2 = nn.Sequential(
            nn.Conv2d(1024,512,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(512,256,kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.sec_conv2 = nn.Sequential(
            nn.Conv2d(512,256,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,64,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        
            
            #nn.Conv2d(128,out_chnl,kernel_size=1, stride=1, padding=0, bias=False),  
            #nn.Sigmoid()
        )
        self.nmODE_classifier = nmODE()
        self.ode_up = ODEBlock(self.nmODE_classifier, adjoint=adjoint)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Sequential(
            nn.Conv2d(64,out_chnl,kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.eval_times = torch.tensor(eval_times).float().cuda()

        
        
    def forward(self,input):
        skip1_cat = self.input(input)
        x = self.downsample1(skip1_cat)
        skip2_cat = self.sec_conv(x)
        x = self.downsample2(skip2_cat)
        skip3_cat = self.third_conv(x)
        x = self.downsample3(skip3_cat)
        x = self.four_conv(x)
        #x = x - skip4_sub
        x = self.upsample4(x)
        x = torch.cat( (x, skip3_cat), dim=1)
        x = self.third_conv2(x)
        #x = x - skip3_sub
        x = self.upsample3(x)
        x = torch.cat( (x, skip2_cat), dim=1)
        x = self.sec_conv2(x)
        #x = x - skip2_sub
        x = self.upsample2(x)
        x = torch.cat( (x, skip1_cat), dim=1)
        x = self.output_conv(x)
        self.nmODE_classifier.fresh(x)
        x = self.ode_up(torch.zeros_like(x), self.eval_times)
        x = self.output(x)
        
        
        return x,skip1_cat,skip2_cat,skip3_cat
    
    
class att_Unet2(nn.Module):
    def __init__(self,in_chnl=5,out_chnl=1,ifode = True, tol=1e-3, adjoint=False,eval_times = (0, 1)):
        super(att_Unet2,self).__init__()
        self.ifode = ifode
        
        self.input = nn.Sequential(
            nn.Conv2d(in_chnl,128,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.sec_conv = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(256,512,kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.third_conv = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(512,1024,kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.four_conv = nn.Sequential(
            nn.Conv2d(1024,1024,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024,1024,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(2048,512,kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.third_conv2 = nn.Sequential(
            nn.Conv2d(1536,512,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(512,256,kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.sec_conv2 = nn.Sequential(
            nn.Conv2d(768,256,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(384,128,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,64,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Conv2d(128,out_chnl,kernel_size=1, stride=1, padding=0, bias=False),  
            #nn.Sigmoid()
        
            
            #nn.Conv2d(64,out_chnl,kernel_size=1, stride=1, padding=0, bias=False),  
            #nn.Sigmoid()
        )
        self.attention = ptv_attention(in_chnl = 6)
        self.nmODE_classifier = nmODE()
        self.ode_up = ODEBlock(self.nmODE_classifier, adjoint=adjoint)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Sequential(
            nn.Conv2d(64,out_chnl,kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.eval_times = torch.tensor(eval_times).float().cuda()
      
        
    def forward(self,input,f1,f2,f3,mask):
        weight = self.attention(input * mask)
        
        skip1_cat = self.input(input)
        x = self.downsample1(skip1_cat)
        skip2_cat = self.sec_conv(x)
        x = self.downsample2(skip2_cat)
        skip3_cat = self.third_conv(x)
        x = self.downsample3(skip3_cat)
        x = self.four_conv(x)
        x = torch.cat( (x, weight), dim=1)
        x = self.upsample4(x)
        feature3 = torch.cat( (x, f3), dim=1)
        x = torch.cat( (x, feature3), dim=1)
        x = self.third_conv2(x)
        x = self.upsample3(x)
        feature2 = torch.cat( (x, f2), dim=1)
        x = torch.cat( (x, feature2), dim=1)
        x = self.sec_conv2(x)
        x = self.upsample2(x)
        feature1 = torch.cat( (x, f1), dim=1)
        x = torch.cat( (x, feature1), dim=1)
        x = self.output_conv(x)
        self.nmODE_classifier.fresh(x)
        x = self.ode_up(torch.zeros_like(x), self.eval_times)
        x = self.output(x)
        
        return x    
    
    

    
class CascadeModel3(nn.Module):
    def __init__(self):
        super(CascadeModel3,self).__init__()
        self.dose_model = att_Unet1(in_chnl=5,out_chnl=1,ifode=False,adjoint=False)
        self.fluence_model = att_Unet2(in_chnl=6,out_chnl=1,ifode=False,adjoint=False)
        
    def forward(self,bev,ptv):
        dose,f1,f2,f3 = self.dose_model(bev)
        bd = torch.cat([bev, dose], dim=1)
        fluence = self.fluence_model(bd,f1,f2,f3,ptv)
     
         
        return dose, fluence
    
class ptv_attention(nn.Module):
    def __init__(self,in_chnl):
        super(ptv_attention,self).__init__()
        self.get_attention = nn.Sequential(
            nn.Conv2d(in_chnl,128,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128,256,kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256,512,kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512,1024,kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            
            nn.Conv2d(1024,1024,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024,1024,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            #nn.Sigmoid()
        )
    def forward(self,mask):
        weight = self.get_attention(mask)
        
        return weight