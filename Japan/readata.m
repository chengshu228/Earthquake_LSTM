clear all;

clc;


fid=fopen('C:\Users\cshu\Desktop\shi\Japan\catalog\JMA-alltype-cata.dat','w');

fraw=fopen('C:\Users\cshu\Desktop\shi\Japan\catalog\JMAcatalog.dat','r');



while feof(fraw)~=1
    ind=0;
    info=fgetl(fraw);
    year=[info(2:5) ' '];
    month=[info(6:7) ' '];
    day=[info(8:9) ' '];
    hour=[info(10:11) ' '];
    min=[info(12:13) ' '];
    sec=[info(14:17) ' '];
    if(sum(sec==' ')==5)
        sec='  0  ';
    end
    
    lat=[info(22:24) ' '];
    lat1=[info(25:28) ' '];
    
    lon=[info(33:36) ' ' ];
    lon1=[info(37:40) ' '];
    
    if(lat(1:2)=='- ')
        lat(1:2)='-0';
    end
    
    if(lon(1:3)=='-  ')
        lon(1:3)='-00';
    elseif(lon(1:2)=='- ')
        lon(1:2)='-0';
    end

    if(sum(lat1==' ')==5&&sum(lon1==' ')==5)
        ind=1;
    end
    if(sum(lat1==' ')==5)
        lat1='  0  ';
    end
    if(sum(lon1==' ')==5)
        lon1='  0  ';
    end
    
    dep=[info(45:49) ' '];
    if(dep(4)==' ')
        dep(4)='0';
    end
    error=[info(50:52) ' '];
    if(sum(error==' ')==4)
        error='0  ';
    end
    
    mag=[info(53:54)];
    if(mag(1)=='A')
        mag=['-1' mag(2)];
    elseif(mag(1)=='B')
        mag=['-2' mag(2)];
    elseif(mag(1)=='C')
        mag=['-3' mag(2)];
    end
    if(sum(mag==' ')==2)
        mag='-100';
    end
    type=[' ' info(61)];
    if(sum(type==' ')==2)
        type=' 0';
    end
    new=[year month day hour min sec lat lat1 lon lon1 dep error mag type];
    if(ind==0)
      fprintf(fid,'%s\n',new);
    end
end

fclose(fid);
fclose(fraw);
