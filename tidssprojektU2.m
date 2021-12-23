d66=load('climate66.dat');
d67=load('climate67.dat');
d68=load('climate68.dat');
d69=load('climate69.dat');
d70=load('climate70.dat');
d71=load('climate71.dat');
d72=load('climate72.dat');
d73=load('climate73.dat');
d=[d66;d67;d68;d69;d70;d71;d72;d73];

jun=find(d70(:,2)==6);
jun=jun(110:end);
dec=find(d70(:,2)==12);

%%
X=d70([jun;find(d70(:,2)==7); find(d70(:,2)==8)],6);
dat=d70([jun;find(d70(:,2)==7); find(d70(:,2)==8)],8);
data=dat(1:7*24*8);
Xdata=X(1:7*24*8);
val=dat(7*24*8+1:7*24*10);
Xval=X(7*24*8+1:7*24*10);
test=dat(7*24*10+1:7*24*11);
Xtest=X(7*24*10+1:7*24*11);
% test2=dec(1:7*24,8);
% Xtest2=dec(1:7*24,6);

 z=iddata(data,Xdata);
% plot(z);

meanX=mean(Xdata);
Xdata=Xdata-meanX;
analyzets(Xdata,50,0.05,1)
ARX=arx(Xdata,5); 
Xpw=filter(ARX.A,1,Xdata);
analyzets(Xpw,50,0.05,2)

meandata=mean(data);
data=data-meandata;
datapw=filter(ARX.A,1,data);
M = 50;
figure(3)
stem(-M:M,xcorr(datapw,Xpw,M,'coeff')); %b=0, r=2, s=0 
title('Crosscorrelationfunction'),xlabel('Lag ' )
hold on
plot(-M:M,2/sqrt(length(datapw))*ones(1,2*M+1), '--')
plot(-M:M,-2/sqrt(length(datapw))*ones(1,2*M+1) , '--' )

A2=[1 0 0]; %F
B=[1];
BJ=idpoly([1],[B],[],[],[A2]);
zpw=iddata(datapw,Xpw);
BJ=pem(zpw,BJ); 
present(BJ)
figure(4)
resid(BJ,zpw,50);


E=data-filter(BJ.b,BJ.f,Xdata);
analyzets(E,50,0.05,5)
ARE=idpoly([1 0 0 0 0 0],[],[1]);
% ARE.Structure.c.Free=[ zeros(1,24) 1];
ARE=pem(E,ARE);
analyzets(filter(ARE.A,1,E),50,0.05,6)
present(ARE)

model=idpoly(1,B,ARE.C,ARE.A,A2);
model=pem(z,model);
present(model)
figure(7)
resid(model,z,50)
%% prediction 
val=dat(7*24*8+1:7*24*10);
val=val-meandata;
Xval=X(7*24*8+1:7*24*10);
Xval=Xval-meanX;
pred=[[data; val] , [Xdata; Xval]];

k=1;
[AS,CS]=equalLength(conv(model.F,model.D),conv(model.F,model.C));
[Fk,Gk]=deconv(conv([1 zeros(1,k-1)],CS),AS); %arma-delen
BF=conv(conv(model.D,model.B),model.F);
[CS,BF]=equalLength(CS,BF);
[Fhat,Ghat]=deconv(conv([1,zeros(1,k-1)],BF),CS);

yhatk=filter(Ghat,CS,pred(:,2))+filter(Gk,CS,pred(:,1))+filter(Fhat,1,pred(:,2));
yhatk=yhatk(end-length(val)+1:end);

figure(1)
plot(val)
hold on
plot(yhatk(k+1:end),'r-')
%% U3 - Kalmanfilter

b=conv(model.b,model.d);

sige=0.5;
sigv=1;
% A1A2=conv(model.d,model.f);
% A1A2=A1A2(2:end);

% Data  length
N =length(data);
% e=sqrt(sige)*randn(N,1);
% v=sqrt(sigv)*randn(N,1);
% y=filter(b,1,u)+filter(1,[1 -1],e)+filter(1,1,v);
% xt=data-b*Xdata-v;
% Define  the  s t a t e  space  equations
A=eye(length(b)+1);
Re=[50 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    ];% Hidden  s t a t e  noise  covariance  matrix
Rw=1*sigv^2;  % Observtion  variance

%usually C should  be  set  here  to ,
%but  in  t h i s  case C i s  a  function  of  time .
% Set some  i n i t i a l  values
Rxx1=eye(length(b)+1);% I n i t i a l  variance
xtt1=[0 ones(1,length(b))]';% I n i t i a l  s t a t e

% Vector  to  store  values  in
xsave=zeros(length(b)+1,N);

% Kalman  f i l t e r .  Start  from k=3,% since we need  old  values  of  y .
for k=length(b)+1:N
    % C is ,  in  our  case ,  a  function  of  time .
    C=[1 flip(Xdata(k-length(b)+1:k))'];
    % Update
    Ryy=C*Rxx1*C'+Rw;
    Kt=Rxx1*C'*1/Ryy;
    xtt=xtt1+((data(k) -C*xtt1)*Kt);
    Rxx=(eye(size(Kt*C))-Kt*C)*Rxx1;
    % Save
    xsave(:,k)=xtt;
    % Predict
    Rxx1=A*Rxx*A'+Re;
    xtt1=A*xtt;
end


figure(1)
plot(data-xsave(1,:)')
analyzets(data-xsave(1,:)',50,0.05,2)
figure(3)
plot(data,'b-');
hold on
plot(xsave(1,:)','r-')