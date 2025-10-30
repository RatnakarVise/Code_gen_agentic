@AccessControl.authorizationCheck: #CHECK
@ObjectModel: {
   usageType: {
      dataClass: #TRANSACTIONAL,
      serviceQuality: #C,
      sizeCategory: #XL
   }
}
@EndUserText.label: 'Material Details with Description'
@Metadata.allowExtensions: true
@VDM.viewType: #CONSUMPTION
@Search.searchable: true
define view entity ZMM_I_MaterialDetl as select from mara as _mara
    association [0..1] to makt as _makt
        on _makt.matnr = _mara.matnr
{
    key _mara.matnr,
    key _makt.spras,
        _mara.ersda,
        _mara.mtart,
        _mara.matkl,
        _makt.spras,
        _makt.maktx
}