@EndUserText.label: 'Value Help for Material'
@ObjectModel.dataCategory: #VALUE_HELP
@AccessControl.authorizationCheck: #NOT_REQUIRED
define view entity Z_I_MaterialVH
  as select from I_Material as Material
{
  key Material as Material,
      MaterialText
}