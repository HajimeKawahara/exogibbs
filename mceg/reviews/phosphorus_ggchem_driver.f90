! Shared-input independent solver check using unmodified GGChem SMCHEM16.
! NASA9 is evaluated here; per-temperature formation constants enter fit 1.
program phosphorus_reference
  use CHEMISTRY, only: NELM,NMOLE,charge,el,cmol,fit,natom,a,m_kind,m_anz, &
       catm,H,He,C,N,O,P,S,elnum,STOImax,Ncmax,NewFullIt,NewBackIt, &
       NewBackFac,NewFastLevel,NewPreMethod
  implicit none
  integer, parameter :: qp = selected_real_kind(33,4931), maxseg=8
  real(qp), parameter :: bk=1.380662e-16_qp, bar=1.e6_qp
  integer :: ne,nm,ncase,ii,jj,kk,nn,ic,it,idx
  integer, allocatable :: nseg(:),composition(:,:)
  real(qp), allocatable :: bounds(:,:,:),coef(:,:,:),abund(:),mono(:),molecules(:),g(:)
  real(qp) :: total,target,pressure,error,nHq,kT,previous_nH,previous_residual,residual,slope
  real(8) :: temp,nH
  logical :: remember
  integer :: verbosity
  character(256) :: input_file,output_file
  character(20) :: name
  call get_command_argument(1,input_file)
  call get_command_argument(2,output_file)
  open(10,file=trim(input_file),status='old')
  read(10,*) ne,nm,ncase
  NELM=ne; NMOLE=nm; charge=.false.; el=0
  allocate(cmol(nm),fit(nm),natom(nm),a(nm,0:13),m_kind(0:6,nm),m_anz(6,nm))
  allocate(nseg(ne+nm),bounds(2,maxseg,ne+nm),coef(9,maxseg,ne+nm))
  allocate(composition(ne,nm),abund(ne),mono(ne),molecules(nm),g(ne+nm))
  do ii=1,ne
    read(10,*) catm(ii),abund(ii),nseg(ii)
    select case(trim(catm(ii)))
    case('H'); H=ii; elnum(ii)=1
    case('He'); He=ii; elnum(ii)=2
    case('C'); C=ii; elnum(ii)=6
    case('N'); N=ii; elnum(ii)=7
    case('O'); O=ii; elnum(ii)=8
    case('P'); P=ii; elnum(ii)=15
    case('S'); S=ii; elnum(ii)=16
    case default; error stop 'Unsupported element in reference driver'
    end select
    do kk=1,nseg(ii)
      read(10,*) bounds(:,kk,ii),coef(:,kk,ii)
    enddo
  enddo
  abund=abund/abund(H)
  STOImax=0; m_kind=0; m_anz=0; a=0; fit=1
  do ii=1,nm
    idx=ne+ii
    read(10,*) cmol(ii),composition(:,ii),nseg(idx)
    natom(ii)=sum(composition(:,ii))
    nn=0
    do jj=1,ne
      if(composition(jj,ii)==0) cycle
      nn=nn+1
      m_kind(nn,ii)=jj; m_anz(nn,ii)=composition(jj,ii)
      STOImax(jj)=max(STOImax(jj),composition(jj,ii))
    enddo
    m_kind(0,ii)=nn
    do kk=1,nseg(idx)
      read(10,*) bounds(:,kk,idx),coef(:,kk,idx)
    enddo
  enddo
  Ncmax=maxval(STOImax)
  NewFullIt=.true.; NewBackIt=5; NewBackFac=1.e10
  NewFastLevel=1; NewPreMethod=2
  open(11,file=trim(output_file),status='replace')
  write(11,*) ne,nm,ncase
  do ic=1,ncase
    read(10,*) temp,pressure
    do ii=1,ne+nm
      idx=0
      do kk=1,nseg(ii)
        if(temp>=bounds(1,kk,ii).and.temp<=bounds(2,kk,ii)) then
          idx=kk
        endif
      enddo
      if(idx==0) error stop 'Temperature outside source polynomial intervals'
      g(ii)=nasa_g(real(temp,qp),coef(:,idx,ii))
    enddo
    do ii=1,nm
      ! GGChem fit1 is lnK in cgs pressure units (dyn/cm2).
      a(ii,0)=real(-g(ne+ii)+sum(composition(:,ii)*g(:ne))+(1-natom(ii))*log(bar),8)
    enddo
    target=pressure*bar
    kT=bk*real(temp,qp)
    nHq=target/(sum(abund)*kT)
    previous_nH=nHq; previous_residual=0
    do it=1,100
      nH=real(nHq,8); remember=.false.; verbosity=-1
      call SMCHEM16(nH,temp,abund,mono,molecules,remember,verbosity)
      total=sum(mono)+sum(molecules)
      error=total*kT/target-1
      if(abs(error)<1.e-12_qp) exit
      residual=log(total*kT/target)
      slope=1
      if(it>1.and.abs(log(nHq/previous_nH))>1.e-20_qp) &
        slope=max(.1_qp,min(2._qp,(residual-previous_residual)/log(nHq/previous_nH)))
      previous_nH=nHq; previous_residual=residual
      nHq=nHq*exp(-residual/slope)
    enddo
    if(it>100) error stop 'Pressure closure failed'
    write(11,'(4ES27.18,I6)') temp,pressure,real(nH,qp),error,it
    write(11,'(*(ES27.18,1X))') log(mono)
    write(11,'(*(ES27.18,1X))') log(molecules)
    write(11,'(*(ES27.18,1X))') real(a(:,0),qp)-(1-natom(:))*log(bar)
  enddo
  close(10);close(11)
contains
  real(qp) function nasa_g(t,c) result(value)
    real(qp),intent(in) :: t,c(9)
    real(qp) :: enthalpy,entropy
    enthalpy=-c(1)/t**2+c(2)*log(t)/t+c(3)+c(4)*t/2+c(5)*t**2/3 &
             +c(6)*t**3/4+c(7)*t**4/5+c(8)/t
    entropy=-c(1)/(2*t**2)-c(2)/t+c(3)*log(t)+c(4)*t+c(5)*t**2/2 &
            +c(6)*t**3/3+c(7)*t**4/4+c(9)
    value=enthalpy-entropy
  end function
end program
