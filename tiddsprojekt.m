d66=load('climate66.dat');
d67=load('climate67.dat');
d68=load('climate68.dat');
d69=load('climate69.dat');
d70=load('climate70.dat');
d71=load('climate71.dat');
d72=load('climate72.dat');
d73=load('climate73.dat');
 d=[d66;d67;d68;d69;d70;d71;d72;d73];
%  plot(d(:,8))

%  figure 
%  plot(d68([find(d68(:,2)==6); find(d68(:,2)==7); find(d68(:,2)==8); find(d68(:,2)==9)],8))
 
jun=find(d70(:,2)==6);
jun=jun(110:end);

% figure

% plot(d70(jun,8))
% figure
% plot(d70([jun; find(d70(:,2)==7); find(d70(:,2)==8)],8))
%% A) create model
dat=d70([jun;find(d70(:,2)==7); find(d70(:,2)==8)],8);
data=dat(1:7*24*8);
val=dat(7*24*8+1:7*24*10);
test=dat(7*24*10+1:7*24*11);


% analyzets(data,50,0.05,1)
%data=sqrt(data);
meandatasqrt=mean(data);
data=data-meandatasqrt;
analyzets(data,50,0.05,2)
figure(3) 
%%plot([data+meandata; val]- mean([data+meandata;val]))


A24=[1 zeros(1,23) -1];
sdata=filter(A24,1,data); %remove seasonal dependency
analyzets(sdata,50,0.05,4)

% analyzets(data,50)
% nest=floor(2/3*length(data));
% for i=1:100
% yest=iddata(data(1:nest));
% yval=iddata(data(nest+1:end));
% V=arxstruc(yest,yval,[1:10]');
% norder(i)=selstruc(V,0);
% naic(i)=selstruc(V,'aic'); %Aikaikes Information Criteria
% end
% figure(10)
% histogram(naic)
arp=arx(sdata,5);

model=idpoly([1 1 1 1 1 1],[],[1 1 1 0 1 0 0 0 0 0 0 0 0 zeros(1,11) -1]);
%model.Structure.a.Free=[ 0 1 1 1 1 1 1 1 zeros(1,16) 1];
model.Structure.c.Free=[0 1 1 1 1 1 1 0 0 0 0 0 0 zeros(1, 11) 1];
model=pem(iddata(sdata),model);
eha=filter(model.a,model.c,sdata);

analyzets(eha,50,0.05,5)
figure(6) 
resid(model,sdata,50)
present(model)
model.a=conv(model.a,A24);
% figure(7)
% resid(model,val,50) 
% analyzets(filter(model.a,model.c,val),50,0.05,8)
%% A) predict 
val=dat(7*24*8+1:7*24*10);
test=dat(7*24*10+1:7*24*11);
k=26;

%val=sqrt(val);
sqrtval=mean(val);
val=val-meandatasqrt;
pred=[data; val];
plot(pred)
[AS,CS]=equalLength(model.a,model.c);
e=filter(AS,model.c,pred);
sig=var(e);

[Fk, Gk]=deconv(conv([1 zeros(1,k-1)],CS),AS);
yhatk=filter(Gk,CS,pred);

yhatk=yhatk(end-length(val)+1:end);

vt=sum(Fk.^2);
v=var(val-yhatk);
c=1.96*sqrt(vt)*sig^2;

analyzets(val-yhatk,50,0.05,2)
 
figure 
plot(val)
hold on 
plot(yhatk)
hold on 
figure
plot(val-yhatk)
%% naive for k=1

yhatk1=[0; val];
figure
plot(1:length(val),val)
hold on
plot(yhatk1,'r-') %prediction


analyzets(val-yhatk1(1:end-1),50,0.05,2)

figure
plot(val-yhatk1(1:end-1))
%% naive for k=7
yhatk7=[zeros(24,1); val ];

figure
plot(val)
hold on
plot(yhatk7,'r-') %prediction


analyzets(val-yhatk7(1:end-24),50,0.05,2)

figure
plot(val-yhatk7(1:end-24))
%% naive k=26 1.0
yhatk26=[zeros(48,1); val ];

figure
plot(val)
hold on
plot(yhatk26,'r-') %prediction


analyzets(val-yhatk26(1:end-48),50,0.05,2)

figure
plot(val-yhatk26(1:end-48))
%% naive k=26 superior version
yhatk26sv=[zeros(2,1); val ];

figure
plot(val)
hold on
plot(yhatk26sv,'r-') %prediction


analyzets(val-yhatk26sv(1:end-2),50,0.05,5)

figure
plot(val-yhatk26sv(1:end-2))
