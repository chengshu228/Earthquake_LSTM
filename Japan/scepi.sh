#!/bin/sh
gmt gmtset FORMAT_GEO_MAP D

cpt0=mby.cpt
cpt1=ETOPO1-sc.cpt
grdfile=etopo1.grd

eqinput=sc201912-M2.5.etas
output=sc.eps

region='-R122W/114W/32N/37N'
s=32
n=37
w=-122
e=-114
mag=3.0
mmag=7.0

gmt psbasemap $region -Xc -Ba2f1dSWEN  -JM6i -P -K -Y10 >$output

gmt grdimage $grdfile -R -JM -C$cpt0 -Q -K -O >>$output
gmt pscoast -R -JM -O -K -Dh -Gc >> $output
gmt grdimage $grdfile -R -JM -C$cpt1 -O -K -B >>$output
gmt pscoast -R -JM -O -Q -B -K >> $output
gmt psscale -D3i/-0.35i/4i/0.15ih -C$cpt1 -B500::/:m: -O -K >>$output
#pscoast -JM -R -W0.25p,black -K -O >> $output
awk '{if($2>=x1 && $2<=x2 && $3>=y1 && $3<=y2 && $4>=m && $8>=1980 ) \
print $2,$3,($4-2.5)/15}' x1=$w x2=$e y1=$s y2=$n m=$mag $eqinput | \
gmt psxy -R -JM -W0.5p,black -Sc -K -O  >> $output


#gmt psxy -R -JM -K -W1,deeppink,-. -O -L <<EOF >> $output
#130.819000000000	32.7815000000000
#131.050400000000	32.9131500000000
#EOF
gmt psxy -R -JM -K -W1,white,-- -O -L <<EOF >> $output
 -119.0 33.0
 -115.0 33.0
 -117.0 37.0
 -120.8 37.0
 -121.5 34.5
-119.0 33.0
EOF

gmt psxy -R -JM -K -W0.8,red -O -L <<EOF >> $output
-116.845555555556	34.6673352435530
-116.695185185185	34.7189111747851
-116.487777777778	34.5865329512894
-116.352962962963	34.2100286532951
-116.187037037037	33.9297994269341
-116.347777777778	33.8747851002865
-116.503333333333	34.1670487106017
-116.492962962963	34.2100286532951
-116.570740740741	34.4128939828080
-116.747037037037	34.5934097421204
>
-118.777534562212	34.3326530612245
-118.677304147465	34.4644314868805
-118.495276497696	34.4317784256560
-118.354723502304	34.3128279883382
-118.496428571429	34.1600583090379
-118.646198156682	34.1938775510204
>
-116.501201646091	34.8523361877160
-116.347190123457	34.9067446080082
-116.252028532236	34.8233955386245
-116.248272153635	34.7192092018949
-116.194430727023	34.5235704140360
-115.986577777778	34.2029080221016
-116.041671330590	34.1704944951191
-116.233246639232	34.3950739320695
-116.273314677641	34.3997044359242
-116.409796433471	34.7284702096042
>
-117.389318600368	35.4746495327103
-117.592633517495	35.6681074766355
-117.766482504604	35.8973130841121
-117.686924493554	35.9982476635514
-117.523388581952	35.7627336448598
-117.320073664825	35.5882009345794
EOF

gmt psxy -R -JM -K -W0.8,orange,-- -O -L <<EOF >> $output

-117.569832826748       35.6880792227205
-117.673784194529       35.5954783258595
-117.620440729483       35.5290358744395
-117.420744680851       35.7016816143498
-117.600379939210       35.8293348281016
-117.647796352584       35.7702167414051
-117.569832826748       35.6880792227205

EOF
#gmt psxy -R -JM -K -Sa0.2 -W0.1p,orange -O -L -Gorange <<EOF >> $output
#130.808697000000	32.7417240000000
#EOF
#gmt psxy -R -JM -K -Sa0.16 -W0.1p,orange -O -L -Gorange <<EOF >> $output
#130.777656000000	32.7007000000000
#EOF
#gmt psxy -R -JM -K -Sa0.25 -W0.1p,red -O -L -Gred <<EOF >> $output
#130.762972000000	32.7544650000000
#EOF

gmt psmeca -R -JM -Ewhite -Gred -Sz0.4/12 -C -W0.8,white -K -O <<EOF >> $output
-116.437 34.20 15 -0.88 -6.12 7.00 3.81 0.10 7.34 26 -117.0 34.7     
-118.537 34.213 17 1.08 -0.94 -0.14 0.05 -0.40 0.44 26 -118.2 34.4    
-116.265 34.603 15 -0.09 -4.27 4.35 0.69 0.98 3.98 26 -115.9 34.7
-117.504 35.705 13 -0.46 -5.68 6.13 -0.00 -0.56 0.49 25 -117.2 35.8
-117.599 35.770 12 -0.23 -4.11 4.34 0.51 0.49 0.95 26 -117.9 35.6
EOF

#gmt psxy -R -JM -K -Ss0.2 -W0.1p,cyan -O -L -Gcyan <<EOF >> $output
#131.49751 33.27945 
#131.354917 33.26243
#EOF
#gmt psxy -R -JM -K -St0.3 -W0.1p,red -O -L -Gred <<EOF >> $output
#131.084107 32.88692
#EOF

gmt pstext -R -JM -Glightblue -K -O << EOF >> $output
-118.47 34.63 9 0 0 M Northridge
-117.75 34.8 9 0 0 M Landers
-115.72 34.7 9 0 0 M Hector Mine
-118.2 35.38  9 0 0 M Ridgecrest
-117.0 35.8 9 0 0 M Foreshock
EOF

#gmt psxy -R -JM -K -W1,blue,-. -O -L <<EOF >> $output
#131.0899 33.3531
#131.4355 33.0306
#EOF

#gmt psxy -R -JM -K -W1,blue,- -O -L <<EOF >> $output
#130.7212 33.0959
#131.1175 32.7286
#EOF

#gmt pstext -R -JM -Wblue -K -O << EOF >> $output
#131.0 32.73 8 0 1 LM A
#131.26 32.93 8 0 1 LM B
#131.45 33.13 8 0 1 LM C
#EOF
#awk '{print $1,$2}' $coast | \
#psxy -R -JM -W1 -P -K -O -M >> $output

