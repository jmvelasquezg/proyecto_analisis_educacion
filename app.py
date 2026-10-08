"""Dashboard descriptivo de matrícula 2021–2024. Ejecutar: streamlit run app.py."""
from pathlib import Path
from io import BytesIO
import zipfile
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st

ANIOS = [2021, 2022, 2023, 2024]
BASE = Path(__file__).resolve().parent
ARCHIVO = 'Matricula_Consolidada_2021_2024.csv.gz'
TEXTOS = ['Nombre','gestion','dsc_nivel','TipoDiscaIntegrada','DPTO','PROV',
          'DIST','DRE_UGEL','DAREACENSO','CEN_POB','DIRECCION','REGION_NAT','Tipo_coincidencia']
NUMEROS = ['Año','Edad','TotalEstudiantes','tot_atraso','Retirado','PromocionGuiada',
           'Desaprobado','Año_padron_utilizado','NLAT_IE','NLONG_IE','ALTITUD']
COLORES = {'Matricula':'#67D3E8','Atraso_pct':'#FFBD69','Retiro_pct':'#79DBAF'}
PALETA = ['#67D3E8','#B8A1FF','#FFBD69','#79DBAF','#F493B0','#9DB9DB']
ETIQUETAS = {
    'Matricula':'Matrícula','Atraso_pct':'Atraso registrado (%)','Retiro_pct':'Retiro registrado (%)',
    'Atraso':'Estudiantes con atraso','Retirados':'Retiros registrados',
    'dsc_nivel':'Nivel educativo','gestion':'Tipo de gestión','Edad':'Edad registrada',
    'cod_mod':'Código modular','anexo':'Anexo','Nombre':'Institución educativa',
    'DPTO':'Departamento','PROV':'Provincia','DIST':'Distrito','DRE_UGEL':'DRE / UGEL',
    'CODLOCAL':'Código de local','DAREACENSO':'Área','CEN_POB':'Centro poblado',
    'DIRECCION':'Dirección','REGION_NAT':'Región natural','ALTITUD':'Altitud (m)',
    'NLAT_IE':'Latitud','NLONG_IE':'Longitud','Año_padron_utilizado':'Año del padrón',
    'Tipo_coincidencia':'Correspondencia temporal','Servicios':'Servicios educativos',
    'Aprobado':'Aprobados','PromocionGuiada':'Promoción guiada','Desaprobado':'Desaprobados',
    'Retirado':'Retiros registrados','Fallecido':'Fallecidos',
    'RequiereRecuperacion':'Requieren recuperación','Matriculado':'Estado matriculado',
    'PostergaEvaluacion':'Evaluación postergada','Variacion_matricula_pct':'Cambio anual de matrícula (%)',
    'Cambio_atraso_pp':'Cambio de atraso (pp)','Cambio_retiro_pp':'Cambio de retiro (pp)'
}


def preparar_datos(d):
    necesarias = ['Año','cod_mod','anexo','Nombre','dsc_nivel','gestion','Edad',
                  'TotalEstudiantes','tot_atraso','Retirado','DPTO','PROV','DIST',
                  'DRE_UGEL','DAREACENSO','Año_padron_utilizado','Tipo_coincidencia']
    faltantes = sorted(set(necesarias)-set(d.columns))
    if faltantes:
        raise ValueError('Se necesita la base enriquecida. Faltan: '+', '.join(faltantes))
    for c in NUMEROS:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors='raise')
    if d[['Año','cod_mod','anexo','TotalEstudiantes','tot_atraso','Retirado']].isna().any().any():
        raise ValueError('Hay claves o conteos esenciales vacíos. Revisa la base antes de analizar.')
    if set(d['Año'].unique()) != set(ANIOS):
        raise ValueError('La base debe contener los cuatro años: 2021–2024.')
    conteos = d[['TotalEstudiantes','tot_atraso','Retirado']]
    if not np.isfinite(conteos.to_numpy(dtype=float)).all() or conteos.lt(0).any().any():
        raise ValueError('Los conteos deben ser números finitos y no negativos.')
    for c, ancho in [('cod_mod',7),('anexo',1),('CODLOCAL',6),('CODGEO',6)]:
        if c in d:
            d[c] = d[c].astype('string').str.strip().str.replace(r'\.0+$','',regex=True).str.zfill(ancho)
    if not d.cod_mod.str.fullmatch(r'\d{7}').all():
        raise ValueError('Hay códigos modulares que no tienen siete dígitos.')
    for c in TEXTOS:
        if c in d:
            s = d[c].astype('string').str.strip()
            d[c] = s.mask(s.eq('')).fillna('Sin dato').astype('category')
    # Las dos columnas de trayectoria permanecen distintas; no se rellenan con cero.
    if 'Desaprobado' in d and d.loc[d['Año'].le(2022),'Desaprobado'].notna().any():
        raise ValueError('Desaprobado tiene datos en años que no incluyen esa columna en la base original.')
    if 'PromocionGuiada' in d and d.loc[d['Año'].ge(2023),'PromocionGuiada'].notna().any():
        raise ValueError('PromocionGuiada tiene datos en años que no incluyen esa columna en la base original.')
    return d


def leer_datos(ruta):
    if Path(ruta).name != ARCHIVO:
        raise ValueError('El archivo de la aplicación no corresponde a la base configurada.')
    tipos = {c:'string' for c in ['cod_mod','anexo','CODLOCAL','CODGEO']}
    tipos.update({c:'category' for c in TEXTOS})
    return preparar_datos(pd.read_csv(ruta,dtype=tipos,compression='gzip',
                                      low_memory=False,encoding='utf-8-sig'))


@st.cache_resource(show_spinner='Preparando los resultados...',max_entries=1)
def cargar_local(ruta, firma):
    # La base compartida solo se lee; los filtros nunca la modifican.
    return leer_datos(Path(ruta))


def indicadores(d, grupos):
    r = d.groupby(grupos,observed=True,dropna=False).agg(
        Matricula=('TotalEstudiantes','sum'),Atraso=('tot_atraso','sum'),
        Retirados=('Retirado','sum')).reset_index()
    den = r.Matricula.where(r.Matricula.gt(0))
    r['Atraso_pct'] = r.Atraso/den*100
    r['Retiro_pct'] = r.Retirados/den*100
    return r


def filtrar(d, niveles, gestiones, departamento, provincia, distrito, edades, incluir_sin_edad):
    m = d.dsc_nivel.isin(niveles) & d.gestion.isin(gestiones)
    for c,v in [('DPTO',departamento),('PROV',provincia),('DIST',distrito)]:
        if v != 'Todos': m &= d[c].eq(v)
    edad = d.Edad.between(*edades)
    if incluir_sin_edad: edad |= d.Edad.isna()
    return d.loc[m & edad]


def entero(v):
    return f'{v:,.0f}'.replace(',',' ')


def porcentaje(v):
    return 'Sin dato' if pd.isna(v) else f'{v:.2f}%'


def serie_completa(r):
    # Años sin registros en un filtro quedan como huecos, nunca como cero.
    return r.set_index('Año').reindex(ANIOS).reset_index()


def estilo_grafico():
    return {
        'background':'#0E1828', 'padding':{'left':8,'top':16,'right':20,'bottom':8},
        'config':{
            'view':{'stroke':None},
            'axis':{'labelColor':'#C4D0E1','titleColor':'#DCE7F5',
                    'gridColor':'#243348','domainColor':'#31445B',
                    'tickColor':'#31445B','labelFontSize':11,'titleFontSize':12},
            'legend':{'labelColor':'#C4D0E1','titleColor':'#DCE7F5','labelFontSize':11},
            'title':{'color':'#EAF1FA','fontSize':15,'anchor':'start'}
        }
    }


def grafico(obj):
    datos,spec=obj
    st.vega_lite_chart(datos,spec,width='stretch',theme=None)


def linea(r, metrica, x='Año', dimension=None):
    grupos=r.groupby(dimension,observed=True,dropna=False) if dimension else [('Serie',r)]
    partes=[]
    for categoria,g in grupos:
        p=serie_completa(g) if x=='Año' else g.sort_values(x).copy()
        # Separar los tramos impide unir años a través de un dato ausente.
        p['Tramo']=p[metrica].isna().cumsum().astype(str)
        if dimension: p[dimension]=str(categoria)
        partes.append(p)
    datos=pd.concat(partes,ignore_index=True)
    datos['Valor']=datos[metrica]/1_000_000 if metrica=='Matricula' and x=='Año' else datos[metrica]
    datos['Etiqueta']=datos[metrica].map(lambda v: '' if pd.isna(v) else
        f'{v/1_000_000:.2f} M' if metrica=='Matricula' and x=='Año' else
        entero(v) if metrica=='Matricula' else f'{v:.2f}%')
    tooltip=[{'field':x,'type':'ordinal','title':'Año' if x=='Año' else 'Edad registrada'}]
    if dimension: tooltip.append({'field':dimension,'type':'nominal','title':ETIQUETAS.get(dimension,dimension)})
    tooltip += [{'field':c,'type':'quantitative','title':ETIQUETAS[c],'format':',.0f' if c=='Matricula' else '.2f'}
                for c in ['Matricula','Atraso_pct','Retiro_pct']]
    enc={
        'x':{'field':x,'type':'ordinal','sort':ANIOS if x=='Año' else 'ascending',
             'axis':{'title':None if x=='Año' else 'Edad registrada (años)','labelAngle':0}},
        'y':{'field':'Valor','type':'quantitative','scale':{'zero':True},
             'axis':{'title':'Millones de matrículas' if metrica=='Matricula' and x=='Año' else ETIQUETAS[metrica]}},
        'detail':{'field':'Tramo','type':'nominal'}, 'tooltip':tooltip
    }
    color={'field':dimension,'type':'nominal','scale':{'range':PALETA},
           'legend':{'title':None,'orient':'bottom','columns':1}} if dimension else {'value':COLORES[metrica]}
    enc['color']=color
    spec=estilo_grafico()
    spec.update({'height':290,'encoding':enc,'layer':[
        {'mark':{'type':'line','strokeWidth':3,'point':{'filled':True,'size':65}}}
    ]})
    if dimension is None and x=='Año':
        spec['layer'].append({'mark':{'type':'text','dy':-16,'fontSize':12,'fontWeight':600},
                              'encoding':{'text':{'field':'Etiqueta'},'color':{'value':'#EAF1FA'}}})
    return datos,spec


def dibujar_serie(r,metrica,dimension=None):
    return linea(r,metrica,'Año',dimension)


def barras(r,categoria,metrica):
    t=r.sort_values(metrica,ascending=False).copy()
    t['Etiqueta']=t[metrica].map(lambda v:entero(v) if metrica=='Matricula' else porcentaje(v))
    maximum=t[metrica].max()
    spec=estilo_grafico()
    spec.update({'height':max(220,len(t)*38),'encoding':{
        'y':{'field':categoria,'type':'nominal','sort':t[categoria].astype(str).tolist(),
             'axis':{'title':None,'labelLimit':250}},
        'x':{'field':metrica,'type':'quantitative','scale':{'domain':[0,max(float(maximum)*1.25,1)]},
             'axis':{'title':ETIQUETAS[metrica]}},
        'tooltip':[{'field':categoria,'type':'nominal','title':ETIQUETAS.get(categoria,categoria)}]+
                  [{'field':c,'type':'quantitative','title':ETIQUETAS[c],
                    'format':',.0f' if c in ['Matricula','Atraso','Retirados'] else '.2f'}
                   for c in ['Matricula','Atraso','Retirados','Atraso_pct','Retiro_pct']]
    },'layer':[
        {'mark':{'type':'bar','color':COLORES[metrica],'cornerRadiusEnd':5,'size':22}},
        {'mark':{'type':'text','align':'left','dx':8,'color':'#EAF1FA','fontSize':12},
         'encoding':{'text':{'field':'Etiqueta'}}}
    ]})
    return t,spec


def tabla(r):
    st.dataframe(r.rename(columns=ETIQUETAS).round(3),hide_index=True,use_container_width=True)


def descarga(r,nombre):
    st.download_button('Descargar esta tabla',r.to_csv(index=False,encoding='utf-8-sig').encode('utf-8-sig'),
                       nombre,'text/csv',key='download_'+nombre)


def hallazgo(r, dimension):
    valid=r.loc[r.Matricula.gt(0)]
    if valid.empty: return 'No hay matrícula positiva para comparar.'
    textos=[]
    for c,conteo,nombre in [('Atraso_pct','Atraso','atraso'),('Retiro_pct','Retirados','retiro')]:
        v=valid.loc[valid[c].idxmax()]
        textos.append(f"**{v[dimension]}** presenta el mayor porcentaje de {nombre}: "
                      f"**{porcentaje(v[c])}** ({entero(v[conteo])} de {entero(v.Matricula)} registros).")
    return ' '.join(textos)


def mapa_tasas(r,fila,columna):
    t=r.copy()
    for c in [fila,columna]:
        t[c]=t[c].map(lambda v:f'{v:g}' if isinstance(v,(float,int,np.number)) else str(v)).astype('string')
    graficos=[]
    for c,colores in [('Atraso_pct',['#192D40','#C2873B','#FFDA9C']),
                     ('Retiro_pct',['#192D40','#2C8975','#A0F3C8'])]:
        base={
            'x':{'field':columna,'type':'ordinal','sort':r[columna].drop_duplicates().sort_values().map(lambda v:f'{v:g}').tolist()
                 if columna=='Edad' else 'ascending',
                 'axis':{'title':None,'labelAngle':-25 if columna!='Edad' else 0,'labelLimit':220}},
            'y':{'field':fila,'type':'nominal','axis':{'title':None,'labelLimit':220}},
            'tooltip':[{'field':fila,'type':'nominal','title':ETIQUETAS.get(fila,fila)},
                       {'field':columna,'type':'ordinal','title':ETIQUETAS.get(columna,columna)},
                       {'field':c,'type':'quantitative','title':ETIQUETAS[c],'format':'.2f'},
                       {'field':'Matricula','type':'quantitative','title':'Matrícula','format':',.0f'}]
        }
        g={'title':ETIQUETAS[c],'height':max(200,t[fila].nunique()*34),
           'encoding':base,'layer':[
               {'mark':{'type':'rect','stroke':'#0E1828','strokeWidth':2},
                'encoding':{'color':{'field':c,'type':'quantitative','scale':{'range':colores,'zero':True},
                            'legend':{'title':'% de matrícula','orient':'bottom'}},
                            'opacity':{'condition':{'test':f'isValid(datum.{c})','value':1},'value':0}}}
           ]}
        if t[columna].nunique()<=4:
            g['layer'].append({'mark':{'type':'text','color':'#FFFFFF','fontSize':12},
                'transform':[{'filter':f'isValid(datum.{c})'}],
                'encoding':{'text':{'field':c,'type':'quantitative','format':'.2f'},
                            'color':{'condition':{'test':f'datum.{c} >= {t[c].max()*0.55 if t[c].notna().any() else 0}', 'value':'#08111F'},'value':'#EAF1FA'}}})
        graficos.append(g)
    spec=estilo_grafico();spec.update({'hconcat':graficos,'spacing':30})
    return t,spec


def paneles_anuales(r, dimension, metrica, edad=False):
    """Cuatro paneles comparables: misma escala, años y categorías."""
    categorias=sorted(r[dimension].dropna().astype(str).unique()) if not edad else None
    fig,axes=plt.subplots(2,2,figsize=(15,9 if not edad else 7),sharex=True,sharey=True)
    fig.patch.set_facecolor('#08111F')
    positivos=r[metrica].dropna()
    limite=float(positivos.max())*1.22 if len(positivos) and positivos.max()>0 else 1
    palette=dict(zip(categorias,PALETA*(len(categorias)//len(PALETA)+1))) if categorias else None
    for anio,ax in zip(ANIOS,axes.flat):
        ax.set_facecolor('#111D30')
        t=r.loc[r['Año'].eq(anio)].copy()
        if edad:
            t=t.set_index('Edad').reindex(range(int(r.Edad.min()),int(r.Edad.max())+1)).reset_index()
            # Matplotlib conserva huecos entre edades sin registros.
            ax.plot(t.Edad,t[metrica],color=COLORES[metrica],linewidth=2.5,marker='o',markersize=4)
            ax.set_ylim(0,min(100,limite) if metrica!='Matricula' else limite)
            ax.set_xticks(range(int(r.Edad.min()),int(r.Edad.max())+1,2))
            ax.set_xlabel('Edad registrada (años)',color='#B6C7DD')
        else:
            t[dimension]=t[dimension].astype(str)
            sns.barplot(data=t,x=metrica,y=dimension,hue=dimension,order=categorias,
                        hue_order=categorias,palette=palette,legend=False,errorbar=None,ax=ax)
            ax.set_xlim(0,limite);ax.set_ylabel('')
            for container in ax.containers:
                ax.bar_label(container,labels=[entero(v) if metrica=='Matricula' else f'{v:.2f}%' for v in container.datavalues],
                             color='#EAF1FA',padding=4,fontsize=8)
            ax.set_xlabel(ETIQUETAS[metrica],color='#B6C7DD')
        if t[metrica].notna().sum()==0:
            ax.text(.5,.5,'Sin registros',transform=ax.transAxes,ha='center',color='#B6C7DD')
        ax.set_title(str(anio),color='#EAF1FA',loc='left',fontweight='bold')
        ax.tick_params(colors='#B6C7DD',labelleft=True,labelbottom=True)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.grid(axis='y' if edad else 'x',color='#2A3A50',alpha=.5)
        ax.set_axisbelow(True)
        if edad:ax.set_ylabel(ETIQUETAS[metrica],color='#B6C7DD')
    fig.tight_layout(pad=2.2)
    return fig


def mostrar_paneles(r,dimension,metrica,edad=False):
    if r.empty:
        st.info('No hay registros para esta comparación.');return
    fig=paneles_anuales(r,dimension,metrica,edad)
    st.pyplot(fig,width='stretch');plt.close(fig)


def narrar_evolucion(serie):
    t=serie.loc[serie.Matricula.gt(0)].sort_values('Año')
    if len(t)<2: return 'Este ámbito tiene información en un solo año. No permite comparar la evolución del periodo.'
    inicio,fin=t.iloc[0],t.iloc[-1]
    cambio=(fin.Matricula/inicio.Matricula-1)*100
    pico=t.loc[t.Matricula.idxmax()]
    sentido='aumentó' if cambio>0 else 'disminuyó' if cambio<0 else 'no cambió'
    return (f"La matrícula **{sentido} {abs(cambio):.2f}%** entre {int(inicio['Año'])} y {int(fin['Año'])}: "
        f"pasó de **{entero(inicio.Matricula)}** a **{entero(fin.Matricula)}**. "
        f"El mayor volumen del periodo se registró en **{int(pico['Año'])}**, con **{entero(pico.Matricula)}** matrículas. "
        f"Durante el mismo periodo, el atraso pasó de **{porcentaje(inicio.Atraso_pct)}** a "
        f"**{porcentaje(fin.Atraso_pct)}**, y el retiro de **{porcentaje(inicio.Retiro_pct)}** a **{porcentaje(fin.Retiro_pct)}**.")


def narrar_comparacion(r,dimension,metrica):
    t=r.loc[r.Matricula.gt(0)&r[metrica].notna()]
    if t.empty: return 'No hay matrícula suficiente para describir esta comparación.'
    mayor=t.loc[t[metrica].idxmax()]
    nombre=f'{mayor[dimension]:g} años' if dimension=='Edad' else str(mayor[dimension])
    if metrica=='Matricula':
        peso=mayor.Matricula/t.Matricula.sum()*100
        return f"**{nombre}** reúne la mayor matrícula: **{entero(mayor.Matricula)}**, equivalente al **{peso:.1f}%** de los grupos comparados."
    conteo='Atraso' if metrica=='Atraso_pct' else 'Retirados'
    indicador='atraso' if metrica=='Atraso_pct' else 'retiro'
    mensaje=(f"El mayor porcentaje de {indicador} aparece en **{nombre}**: **{porcentaje(mayor[metrica])}** "
             f"(**{entero(mayor[conteo])}** casos entre **{entero(mayor.Matricula)}** matrículas).")
    if len(t)>1:
        menor=t.loc[t[metrica].idxmin()]
        nombre_menor=f'{menor[dimension]:g} años' if dimension=='Edad' else str(menor[dimension])
        mensaje+=f" En **{nombre_menor}**, el porcentaje es **{porcentaje(menor[metrica])}**."
    return mensaje


def paquete_tablas(tablas):
    salida=BytesIO()
    with zipfile.ZipFile(salida,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for nombre,t in tablas.items():
            z.writestr(nombre+'.csv',t.to_csv(index=False).encode('utf-8-sig'))
    return salida.getvalue()


def main():
    st.set_page_config(page_title='Perú · Matrícula y trayectoria',page_icon='📚',layout='wide')
    st.markdown('''<style>
    .stApp{background:#08111F;color:#EAF1FA}
    [data-testid="stHeader"]{background:rgba(8,17,31,.94)}
    [data-testid="stSidebar"]{background:#111D30;border-right:1px solid #243348}
    [data-testid="stMetric"]{background:linear-gradient(135deg,#122138,#0E1828);border:1px solid #2B4058;
       border-radius:16px;padding:20px;box-shadow:0 8px 24px #00000022}
    [data-testid="stMetricValue"]{color:#EAF1FA}
    [data-testid="stMetricLabel"]{color:#B6C7DD}
    .eyebrow{color:#67D3E8;font-weight:700;letter-spacing:.16em;font-size:.76rem;margin-bottom:12px}
    .lead{font-size:1.1rem;color:#B6C7DD;max-width:1050px;line-height:1.65}
    h1,h2,h3{color:#EAF1FA;letter-spacing:-.025em}
    [data-testid="stCaptionContainer"]{color:#9DB0C8}
    [data-baseweb="tab-list"]{gap:14px;border-bottom:1px solid #243348}
    [data-baseweb="tab"]{color:#B6C7DD}
    [aria-selected="true"][data-baseweb="tab"]{color:#67D3E8}
    </style>''',unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">EDUCACIÓN EN EL PERÚ · 2021–2024</div>',unsafe_allow_html=True)
    st.title('Matrícula, atraso y retiro escolar en el Perú')
    st.markdown('**¿Cómo cambiaron la matrícula, el atraso y el retiro escolar en el Perú entre 2021 y 2024, según nivel educativo, edad y tipo de gestión?**')
    st.markdown('<p class="lead">El número de matrículas muestra la escala del sistema educativo. '
                'El atraso y el retiro revelan diferencias entre niveles, edades, gestiones y territorios. '
                'Este recorrido permite observar cómo cambió cada indicador entre 2021 y 2024.</p>',unsafe_allow_html=True)
    rutas=[BASE/'Datos'/ARCHIVO,BASE/ARCHIVO]
    ruta=next((p for p in rutas if p.is_file()),None)
    if ruta is None:
        st.error('Los resultados no están disponibles en este momento.');return
    try:
        d=cargar_local(str(ruta),(ruta.stat().st_size,ruta.stat().st_mtime_ns))
    except Exception:
        logging.exception('No se pudo cargar la base del dashboard')
        st.error('Los resultados no están disponibles en este momento.');return

    st.sidebar.header('Explorar los resultados')
    niveles=st.sidebar.multiselect('Nivel educativo',sorted(d.dsc_nivel.unique()),default=sorted(d.dsc_nivel.unique()))
    gestiones=st.sidebar.multiselect('Tipo de gestión',sorted(d.gestion.unique()),default=sorted(d.gestion.unique()))
    departamento=st.sidebar.selectbox('Departamento',['Todos']+sorted(d.DPTO.unique()))
    parcial=d if departamento=='Todos' else d.loc[d.DPTO.eq(departamento)]
    provincia=st.sidebar.selectbox('Provincia',['Todos']+sorted(parcial.PROV.unique()))
    parcial=parcial if provincia=='Todos' else parcial.loc[parcial.PROV.eq(provincia)]
    distrito=st.sidebar.selectbox('Distrito',['Todos']+sorted(parcial.DIST.unique()))
    con_edad=d.Edad.dropna()
    if con_edad.empty:
        edades=(0,0)
    else:
        minimo,maximo=int(con_edad.min()),int(con_edad.max())
        edades=st.sidebar.slider('Edad registrada',minimo,maximo,(minimo,maximo)) if minimo<maximo else (minimo,maximo)
    incluir_sin_edad=st.sidebar.checkbox('Incluir registros sin edad',value=True)
    st.sidebar.caption('Los filtros se aplican a los cuatro años. Un año sin registros se muestra como vacío.')
    f=filtrar(d,niveles,gestiones,departamento,provincia,distrito,edades,incluir_sin_edad)
    if f.empty:
        st.info('No hay registros para esta combinación. Amplía los filtros.');return
    serie=indicadores(f,['Año']).sort_values('Año')
    anios_actuales=sorted(f['Año'].unique().astype(int))
    year=st.sidebar.selectbox('Año para tarjetas, territorio y servicio',anios_actuales,index=len(anios_actuales)-1)
    actual=f.loc[f['Año'].eq(year)]
    seleccion=serie.loc[serie['Año'].eq(year)].iloc[0]
    if seleccion.Matricula<=0:
        st.info('El año seleccionado no tiene matrícula positiva en este ámbito. Elige otro año o amplía los filtros.')
        return
    anterior=serie.loc[serie['Año'].eq(year-1)]
    servicio=actual[['cod_mod','anexo']].drop_duplicates()
    ambito=' / '.join(v for v in [departamento,provincia,distrito] if v!='Todos') or 'Perú'
    st.caption(f'Ámbito: {ambito} · {len(niveles)} niveles · {len(gestiones)} gestiones · '
               f'edad {edades[0]}–{edades[1]}'+(' y sin dato' if incluir_sin_edad else '')+
               f' · tarjetas del año {year}')
    columnas=st.columns(4)
    for col,c,titulo in zip(columnas[:3],['Matricula','Atraso_pct','Retiro_pct'],
                           [f'Matrícula {year}','Atraso registrado','Retiro registrado']):
        valor=entero(seleccion[c]) if c=='Matricula' else porcentaje(seleccion[c])
        delta=None
        if not anterior.empty and pd.notna(anterior.iloc[0][c]) and pd.notna(seleccion[c]):
            diferencia=seleccion[c]-anterior.iloc[0][c]
            delta=(f'{diferencia:+,.0f} registros' if c=='Matricula' else f'{diferencia:+.2f} pp')+f' vs. {year-1}'
        col.metric(titulo,valor,delta=delta,delta_color='off')
    columnas[3].metric('Servicios educativos',entero(len(servicio)))
    st.caption('Los porcentajes indican qué parte de la matrícula registra atraso o retiro. '
               'Cada servicio se identifica por código modular y anexo. Retiro registrado no equivale a deserción.')
    inconsistentes=actual.tot_atraso.gt(actual.TotalEstudiantes)|actual.Retirado.gt(actual.TotalEstudiantes)
    if inconsistentes.any():
        st.warning(f'{entero(inconsistentes.sum())} filas tienen un indicador mayor que la matrícula. '
                   'Revisa los datos antes de interpretar los porcentajes; no se han corregido ni ocultado.')

    st.subheader('Los cuatro años en cifras')
    tabla(serie)

    tabs=st.tabs(['1 · Evolución','2 · Nivel y gestión','3 · Edad','4 · Territorio','5 · Servicio y datos'])
    por_nivel=indicadores(f,['Año','dsc_nivel'])
    por_gestion=indicadores(f,['Año','gestion'])
    por_edad=indicadores(f,['Año','Edad'])
    cruzada=indicadores(f,['Año','dsc_nivel','gestion'])

    with tabs[0]:
        st.header('1. Qué cambió en la matrícula y en las trayectorias')
        st.markdown(narrar_evolucion(serie))
        st.caption('La comparación describe los registros de cada año; no sigue a los mismos estudiantes a lo largo del tiempo.')
        cols=st.columns(3)
        for col,c in zip(cols,['Matricula','Atraso_pct','Retiro_pct']):
            with col:
                st.subheader(ETIQUETAS[c]);grafico(dibujar_serie(serie,c))
        cambios=serie_completa(serie)
        cambios['Variacion_matricula_pct']=cambios.Matricula.pct_change(fill_method=None)*100
        cambios['Cambio_atraso_pp']=cambios.Atraso_pct.diff()
        cambios['Cambio_retiro_pp']=cambios.Retiro_pct.diff()
        with st.expander('Ver cifras y cambios interanuales'):
            tabla(cambios);descarga(cambios,'evolucion_filtrada.csv')

    with tabs[1]:
        st.header('2. Dónde se concentran las diferencias')
        metrica=st.radio('Qué comparar',['Atraso_pct','Retiro_pct','Matricula'],format_func=ETIQUETAS.get,horizontal=True,key='comparacion')
        st.subheader('Nivel educativo · comparación de los cuatro años')
        mostrar_paneles(por_nivel,'dsc_nivel',metrica)
        st.subheader('Tipo de gestión · comparación de los cuatro años')
        mostrar_paneles(por_gestion,'gestion',metrica)
        st.caption('Cada panel corresponde a un año. Las escalas y los colores son iguales para facilitar la comparación. Los porcentajes usan la matrícula de cada grupo.')
        st.subheader('Qué muestran las diferencias año a año')
        for anio in ANIOS:
            nivel=por_nivel.loc[por_nivel['Año'].eq(anio)]
            gestion=por_gestion.loc[por_gestion['Año'].eq(anio)]
            st.markdown(f'**{anio}.** '+narrar_comparacion(nivel,'dsc_nivel',metrica)+' '+narrar_comparacion(gestion,'gestion',metrica))
        st.subheader('Cómo cambia la gestión al comparar el mismo nivel')
        nivel_cruce=st.selectbox('Nivel para comparar gestiones',sorted(f.dsc_nivel.unique()),key='nivel_cruce')
        mismo_nivel=cruzada.loc[cruzada.dsc_nivel.eq(nivel_cruce)]
        mostrar_paneles(mismo_nivel,'gestion',metrica)
        st.caption('Compara gestiones dentro del nivel elegido en 2021, 2022, 2023 y 2024. Las diferencias son descriptivas; no demuestran un efecto causal de la gestión.')
        with st.expander('Ver las cifras por nivel y gestión'):
            tabla(cruzada);descarga(cruzada,'nivel_gestion_filtrado.csv')

    with tabs[2]:
        st.header('3. La edad se interpreta junto con el nivel educativo')
        st.caption('Una edad puede aparecer en distintos niveles educativos. Por eso, una diferencia por edad necesita leerse junto con el nivel y el tamaño del grupo.')
        st.subheader('Qué ocurre con el atraso y el retiro en cada año')
        for anio in ANIOS:
            elegidos=por_edad.loc[por_edad['Año'].eq(anio)&por_edad.Edad.notna()]
            st.markdown(f'**{anio}.** '+narrar_comparacion(elegidos,'Edad','Atraso_pct')+' '+narrar_comparacion(elegidos,'Edad','Retiro_pct'))
        columnas_edad=st.columns(3)
        edades_grafico=por_edad.loc[por_edad.Edad.notna()].copy()
        edades_grafico['Periodo']=edades_grafico['Año'].astype(int).astype(str)
        for col,c in zip(columnas_edad,['Matricula','Atraso_pct','Retiro_pct']):
            with col:
                st.subheader(ETIQUETAS[c])
                grafico(linea(edades_grafico,c,'Edad','Periodo'))
        sin_edad=f.loc[f.Edad.isna()].groupby('Año',observed=True).TotalEstudiantes.sum()
        st.caption('Matrícula sin edad: '+ ' · '.join(f'{a}: {entero(sin_edad.get(a,0))}' for a in ANIOS)+'. Se incluye en los totales si activaste «Incluir registros sin edad», pero no en los gráficos por edad.')
        st.subheader('Edad dentro de un mismo nivel · 2021–2024')
        nivel_edad=st.selectbox('Nivel para comparar edades',sorted(f.dsc_nivel.unique()),key='nivel_edad')
        edades_nivel=indicadores(f.loc[f.dsc_nivel.eq(nivel_edad)],['Año','Edad'])
        edades_nivel=edades_nivel.loc[edades_nivel.Edad.notna()]
        st.caption('Cada panel muestra un año del nivel elegido, con la misma escala. Los huecos representan edades sin registros. El atraso es el registrado en la base; no se calcula a partir de la edad.')
        for metrica_edad in ['Atraso_pct','Retiro_pct']:
            st.subheader(ETIQUETAS[metrica_edad])
            mostrar_paneles(edades_nivel,'Edad',metrica_edad,edad=True)
        st.caption('Un porcentaje alto puede corresponder a pocos estudiantes. Revisa la matrícula de cada edad antes de interpretar la diferencia.')
        with st.expander('Ver cantidades por edad dentro del nivel elegido'):
            tabla(edades_nivel)
        with st.expander('Ver cantidades y denominadores por edad'):
            tabla(por_edad);descarga(por_edad,'edad_filtrada.csv')

    with tabs[3]:
        st.header('4. Las diferencias también tienen una geografía')
        st.caption(f'Comparación territorial de {year}. La ubicación disponible de un servicio puede corresponder a un padrón de otro año.')
        contemporaneos=actual.Año_padron_utilizado.eq(actual['Año'])
        cobertura=actual.loc[contemporaneos,'TotalEstudiantes'].sum()/seleccion.Matricula*100 if seleccion.Matricula else np.nan
        st.markdown(f'**{porcentaje(cobertura)}** de la matrícula del ámbito usa atributos de un padrón del mismo año. ')
        solo_mismo=st.checkbox('Comparar territorios solo con el padrón del mismo año',value=False)
        territorial=actual.loc[contemporaneos] if solo_mismo else actual
        if territorial.empty: st.info('No hay registros territoriales para esta opción.')
        else:
            nivel_territorio=st.selectbox('Agrupar por',['Departamento','Provincia','Distrito','DRE / UGEL','Área'])
            dimensiones={'Departamento':['DPTO'],'Provincia':['DPTO','PROV'],
                         'Distrito':['DPTO','PROV','DIST'],'DRE / UGEL':['DPTO','DRE_UGEL'],
                         'Área':['DAREACENSO']}[nivel_territorio]
            terr=indicadores(territorial,dimensiones)
            terr['Territorio']=terr[dimensiones].astype('string').agg(' / '.join,axis=1)
            c=st.radio('Indicador territorial',['Matricula','Atraso_pct','Retiro_pct'],format_func=ETIQUETAS.get,horizontal=True,key='territorial')
            umbral=st.number_input('Matrícula mínima para aparecer en el gráfico',min_value=0,value=1000,step=100)
            ranking=terr.loc[terr.Matricula.ge(umbral)].nlargest(12,c)
            st.caption('Se muestran hasta 12 grupos. El umbral solo afecta al gráfico; la tabla conserva todos los grupos.')
            if not ranking.empty:
                st.markdown(narrar_comparacion(terr.loc[terr.Matricula.ge(umbral)],'Territorio',c))
                grafico(barras(ranking,'Territorio',c))
            else: st.info('Ningún grupo alcanza el umbral. Reduce la matrícula mínima.')
            tabla(terr);descarga(terr,'territorio_filtrado.csv')
            if st.checkbox('Mostrar ubicación de servicios educativos',value=False):
                if not {'NLAT_IE','NLONG_IE'}<=set(territorial.columns): st.info('La base no incluye coordenadas.')
                else:
                    posiciones=territorial.groupby(['cod_mod','anexo'],observed=True).agg(
                        Matricula=('TotalEstudiantes','sum'),lat=('NLAT_IE','first'),lon=('NLONG_IE','first')).reset_index()
                    validas=posiciones.loc[posiciones.lat.between(-90,90)&posiciones.lon.between(-180,180)&
                                           ~(posiciones.lat.eq(0)&posiciones.lon.eq(0))]
                    visibles=validas.nlargest(3000,'Matricula')
                    if visibles.empty: st.info('No hay coordenadas válidas en el ámbito seleccionado.')
                    else:
                        st.map(visibles[['lat','lon']])
                        st.caption(f'Mapa de {entero(len(visibles))} servicios con mayor matrícula, '
                                   f'de {entero(len(validas))} con coordenadas válidas. '
                                   'La posición es la del padrón utilizado, no necesariamente la ubicación histórica.')
        with st.expander('Alcance de la información territorial'):
            fuentes=actual[['cod_mod','anexo','Año_padron_utilizado','Tipo_coincidencia']].drop_duplicates()
            fuentes=fuentes.groupby(['Año_padron_utilizado','Tipo_coincidencia'],observed=True,dropna=False).size().reset_index(name='Servicios')
            tabla(fuentes)
            st.caption('Esta tabla cuenta servicios, no filas ni estudiantes. Una coincidencia no completa '
                       'automáticamente los atributos vacíos del padrón.')

    with tabs[4]:
        st.header('5. Del panorama general a cada servicio educativo')
        codigo=st.text_input('Código modular (siete dígitos)',placeholder='Ejemplo: 0001506').strip()
        if codigo:
            codigo=codigo.zfill(7)
            consulta=f.loc[f.cod_mod.eq(codigo)]
            if consulta.empty: st.info('No hay registros de ese código en el ámbito de los filtros.')
            else:
                anexos=sorted(consulta.anexo.unique())
                anexo=st.selectbox('Anexo',['Todos']+anexos)
                if anexo!='Todos': consulta=consulta.loc[consulta.anexo.eq(anexo)]
                campos=[c for c in ['Año','cod_mod','anexo','Nombre','dsc_nivel','gestion','CODLOCAL',
                       'DPTO','PROV','DIST','DRE_UGEL','DAREACENSO','CEN_POB','DIRECCION',
                       'REGION_NAT','ALTITUD','NLAT_IE','NLONG_IE','Año_padron_utilizado','Tipo_coincidencia'] if c in consulta]
                tabla(consulta[campos].drop_duplicates())
                resumen_servicio=indicadores(consulta,['Año','cod_mod','anexo'])
                tabla(resumen_servicio);descarga(resumen_servicio,'consulta_servicio.csv')
        st.subheader('Estados de trayectoria disponibles por año')
        estados=[c for c in ['Aprobado','PromocionGuiada','Desaprobado','Retirado','Fallecido',
                             'RequiereRecuperacion','Matriculado','PostergaEvaluacion'] if c in f]
        trayectoria=f.groupby('Año')[estados].sum(min_count=1).reset_index()
        tabla(trayectoria)
        st.caption('Promoción guiada (2021–2022) y desaprobación (2023–2024) son estados diferentes. '
                   'Los datos no disponibles permanecen vacíos, no se interpretan como cero.')
        exports={'evolucion':serie_completa(serie),'nivel':por_nivel,'gestion':por_gestion,
                 'edad':por_edad,'nivel_gestion':cruzada,'estados_trayectoria':trayectoria,
                 'departamentos':indicadores(f,['Año','DPTO']),
                 'fuentes_padron':f[['Año','cod_mod','anexo','Año_padron_utilizado','Tipo_coincidencia']]
                                 .drop_duplicates().groupby(['Año','Año_padron_utilizado','Tipo_coincidencia'],
                                  observed=True,dropna=False).size().reset_index(name='Servicios')}
        st.download_button('Descargar tablas del ámbito (ZIP)',paquete_tablas(exports),
                           'EDA_matricula_filtrado.zip','application/zip')
        with st.expander('Cómo interpretar este dashboard'):
            st.markdown('''- Cada fila del archivo resume un grupo; no representa necesariamente un estudiante.
- Las repeticiones de código modular son esperadas. Los servicios se distinguen por código y anexo.
- Los porcentajes se calculan con conteos agregados y su matrícula, no promediando porcentajes por fila.
- Años sin registros y variables no disponibles permanecen vacíos.
- Los filtros cambian el ámbito de todos los capítulos; la opción de padrón contemporáneo solo afecta al capítulo territorial.
- Los datos de ubicación son los del padrón utilizado. No confirman por sí solos la ubicación histórica.
- Retiro registrado no equivale a deserción. Las diferencias descriptivas no permiten establecer causas.''')
    st.divider()
    st.caption('Una lectura descriptiva de la matrícula escolar: los indicadores muestran diferencias registradas, no sus causas. Retiro no equivale automáticamente a deserción.')


if __name__ == '__main__':
    main()
