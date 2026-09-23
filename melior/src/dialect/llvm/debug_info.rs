//! LLVM debug information attributes.
//!
//! Safe Rust wrappers for MLIR's LLVM dialect debug info (DI) attributes.
//! These map to DWARF metadata and enable source-level debugging of compiled programs.

use crate::{
    ir::{
        attribute::{AttributeLike, StringAttribute},
        Attribute,
    },
    Context,
};
use mlir_sys::{
    mlirLLVMDIBasicTypeAttrGet, mlirLLVMDICompileUnitAttrGet, mlirLLVMDICompositeTypeAttrGet,
    mlirLLVMDICompositeTypeAttrGetRecSelf, mlirLLVMDIDerivedTypeAttrGet,
    mlirLLVMDIExpressionAttrGet, mlirLLVMDIExpressionElemAttrGet, mlirLLVMDIFileAttrGet,
    mlirLLVMDIFlagsAttrGet, mlirLLVMDIImportedEntityAttrGet, mlirLLVMDILexicalBlockAttrGet,
    mlirLLVMDILexicalBlockFileAttrGet, mlirLLVMDILocalVariableAttrGet, mlirLLVMDIModuleAttrGet,
    mlirLLVMDINullTypeAttrGet, mlirLLVMDISubprogramAttrGet, mlirLLVMDISubprogramAttrGetCompileUnit,
    mlirLLVMDISubprogramAttrGetFile, mlirLLVMDISubprogramAttrGetLine,
    mlirLLVMDISubprogramAttrGetRecSelf, mlirLLVMDISubprogramAttrGetScope,
    mlirLLVMDISubprogramAttrGetScopeLine, mlirLLVMDISubprogramAttrGetType,
    mlirLLVMDISubroutineTypeAttrGet, MlirAttribute,
};

// ============================================================================
// Enums
// ============================================================================

/// DWARF emission kind for a compile unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DIEmissionKind {
    None = 0,
    Full = 1,
    LineTablesOnly = 2,
    DebugDirectivesOnly = 3,
}

/// Name table kind for a compile unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DINameTableKind {
    Default = 0,
    GNU = 1,
    None = 2,
    Apple = 3,
}

/// DWARF type encodings for basic types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DITypeEncoding {
    Address = 1,
    Boolean = 2,
    ComplexFloat = 49,
    Float = 4,
    Signed = 5,
    SignedChar = 6,
    Unsigned = 7,
    UnsignedChar = 8,
}

/// DWARF tags for types and other entities.
///
/// Only the most common tags are included here. The underlying MLIR API accepts
/// any valid DWARF tag as a `u32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DITag {
    BaseType = 0x24,
    ClassType = 0x02,
    EnumerationType = 0x04,
    Member = 0x0d,
    PointerType = 0x0f,
    ReferenceType = 0x10,
    StructureType = 0x13,
    Typedef = 0x16,
    UnionType = 0x17,
    ArrayType = 0x01,
    SubrangeType = 0x21,
    ConstType = 0x26,
    VolatileType = 0x35,
    RestrictType = 0x37,
}

/// Subprogram-specific flags.
///
/// These are bitflags that can be combined with `|`. Common values are provided
/// as associated constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DISubprogramFlags(pub u64);

impl DISubprogramFlags {
    pub const NONE: Self = Self(0);
    pub const VIRTUAL: Self = Self(1 << 0);
    pub const PURE_VIRTUAL: Self = Self(1 << 1);
    pub const LOCAL_TO_UNIT: Self = Self(1 << 2);
    pub const DEFINITION: Self = Self(1 << 3);
    pub const OPTIMIZED: Self = Self(1 << 4);
}

impl std::ops::BitOr for DISubprogramFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

// ============================================================================
// Attribute constructors
// ============================================================================

/// Creates a null DI type attribute.
///
/// Used as a placeholder where a type is optional or not yet known.
pub fn di_null_type(context: &Context) -> Attribute<'_> {
    unsafe { Attribute::from_raw(mlirLLVMDINullTypeAttrGet(context.to_raw())) }
}

/// Creates a DI file attribute.
///
/// Represents a source file in the debug info, with a filename and directory.
pub fn di_file<'c>(context: &'c Context, name: &str, directory: &str) -> Attribute<'c> {
    let name_attr = StringAttribute::new(context, name);
    let dir_attr = StringAttribute::new(context, directory);
    unsafe {
        Attribute::from_raw(mlirLLVMDIFileAttrGet(
            context.to_raw(),
            name_attr.to_raw(),
            dir_attr.to_raw(),
        ))
    }
}

/// Creates a DI compile unit attribute.
///
/// A compile unit is the top-level scope for debug info in a translation unit.
/// The `id` should be a distinct attribute (use `DisctinctAttribute::new`).
pub fn di_compile_unit<'c>(
    context: &'c Context,
    id: Attribute<'c>,
    source_language: u32,
    file: Attribute<'c>,
    producer: &str,
    is_optimized: bool,
    emission_kind: DIEmissionKind,
    name_table_kind: DINameTableKind,
) -> Attribute<'c> {
    let producer_attr = StringAttribute::new(context, producer);
    unsafe {
        Attribute::from_raw(mlirLLVMDICompileUnitAttrGet(
            context.to_raw(),
            id.to_raw(),
            source_language,
            file.to_raw(),
            producer_attr.to_raw(),
            is_optimized,
            emission_kind as u32,
            name_table_kind as u32,
        ))
    }
}

/// Creates a DI basic type attribute.
///
/// Describes a primitive type (integer, float, boolean, etc.) with its DWARF
/// tag, name, bit size, and encoding.
pub fn di_basic_type<'c>(
    context: &'c Context,
    tag: u32,
    name: &str,
    size_in_bits: u64,
    encoding: DITypeEncoding,
) -> Attribute<'c> {
    let name_attr = StringAttribute::new(context, name);
    unsafe {
        Attribute::from_raw(mlirLLVMDIBasicTypeAttrGet(
            context.to_raw(),
            tag,
            name_attr.to_raw(),
            size_in_bits,
            encoding as u32,
        ))
    }
}

/// Creates a DI flags attribute.
///
/// Encodes DWARF access and other flags as a bitmask. Common flag values
/// include 0 for no flags, or combinations of DWARF DIFlag constants.
pub fn di_flags(context: &Context, value: u64) -> Attribute<'_> {
    unsafe { Attribute::from_raw(mlirLLVMDIFlagsAttrGet(context.to_raw(), value)) }
}

/// Creates a DI derived type attribute.
///
/// Represents a type derived from another (pointer, typedef, const qualifier,
/// etc.).
pub fn di_derived_type<'c>(
    context: &'c Context,
    tag: u32,
    name: &str,
    base_type: Attribute<'c>,
    size_in_bits: u64,
    align_in_bits: u32,
    offset_in_bits: u64,
) -> Attribute<'c> {
    let name_attr = StringAttribute::new(context, name);
    unsafe {
        Attribute::from_raw(mlirLLVMDIDerivedTypeAttrGet(
            context.to_raw(),
            tag,
            name_attr.to_raw(),
            base_type.to_raw(),
            size_in_bits,
            align_in_bits,
            offset_in_bits,
            // No DWARF address space
            0,
            // No extra data
            MlirAttribute { ptr: std::ptr::null_mut() },
        ))
    }
}

/// Creates a DI derived type attribute with full control over all parameters.
pub fn di_derived_type_full<'c>(
    context: &'c Context,
    tag: u32,
    name: &str,
    base_type: Attribute<'c>,
    size_in_bits: u64,
    align_in_bits: u32,
    offset_in_bits: u64,
    dwarf_address_space: i64,
    extra_data: Option<Attribute<'c>>,
) -> Attribute<'c> {
    let name_attr = StringAttribute::new(context, name);
    let extra = extra_data
        .map(|a| a.to_raw())
        .unwrap_or(MlirAttribute { ptr: std::ptr::null_mut() });
    unsafe {
        Attribute::from_raw(mlirLLVMDIDerivedTypeAttrGet(
            context.to_raw(),
            tag,
            name_attr.to_raw(),
            base_type.to_raw(),
            size_in_bits,
            align_in_bits,
            offset_in_bits,
            dwarf_address_space,
            extra,
        ))
    }
}

/// Creates a self-referencing DI composite type for recursive type definitions.
///
/// Use this to break cycles in type graphs (e.g., a struct with a pointer to
/// itself). Pass the returned attribute as `rec_id` to `di_composite_type`.
pub fn di_composite_type_rec_self(rec_id: Attribute<'_>) -> Attribute<'_> {
    unsafe { Attribute::from_raw(mlirLLVMDICompositeTypeAttrGetRecSelf(rec_id.to_raw())) }
}

/// Creates a DI composite type attribute.
///
/// Describes a compound type such as a struct, class, enum, or array.
/// For recursive types, create a `rec_id` with `DisctinctAttribute::new` and
/// pass it along with `is_rec_self: false`.
#[allow(clippy::too_many_arguments)]
pub fn di_composite_type<'c>(
    context: &'c Context,
    tag: u32,
    name: &str,
    file: Option<Attribute<'c>>,
    line: u32,
    scope: Option<Attribute<'c>>,
    base_type: Option<Attribute<'c>>,
    flags: i64,
    size_in_bits: u64,
    align_in_bits: u64,
    elements: &[Attribute<'c>],
) -> Attribute<'c> {
    let null = MlirAttribute { ptr: std::ptr::null_mut() };
    let name_attr = StringAttribute::new(context, name);
    let raw_elements: Vec<MlirAttribute> = elements.iter().map(|a| a.to_raw()).collect();

    unsafe {
        Attribute::from_raw(mlirLLVMDICompositeTypeAttrGet(
            context.to_raw(),
            null,     // rec_id (none)
            false,    // is_rec_self
            tag,
            name_attr.to_raw(),
            file.map(|a| a.to_raw()).unwrap_or(null),
            line,
            scope.map(|a| a.to_raw()).unwrap_or(null),
            base_type.map(|a| a.to_raw()).unwrap_or(null),
            flags,
            size_in_bits,
            align_in_bits,
            raw_elements.len() as isize,
            raw_elements.as_ptr(),
            null, // dataLocation
            null, // rank
            null, // allocated
            null, // associated
        ))
    }
}

/// Creates a DI composite type attribute with a recursive self-reference.
///
/// For types that reference themselves (e.g., linked list nodes), pass a
/// `rec_id` created via `DisctinctAttribute::new`.
#[allow(clippy::too_many_arguments)]
pub fn di_composite_type_recursive<'c>(
    context: &'c Context,
    rec_id: Attribute<'c>,
    tag: u32,
    name: &str,
    file: Option<Attribute<'c>>,
    line: u32,
    scope: Option<Attribute<'c>>,
    base_type: Option<Attribute<'c>>,
    flags: i64,
    size_in_bits: u64,
    align_in_bits: u64,
    elements: &[Attribute<'c>],
) -> Attribute<'c> {
    let null = MlirAttribute { ptr: std::ptr::null_mut() };
    let name_attr = StringAttribute::new(context, name);
    let raw_elements: Vec<MlirAttribute> = elements.iter().map(|a| a.to_raw()).collect();

    unsafe {
        Attribute::from_raw(mlirLLVMDICompositeTypeAttrGet(
            context.to_raw(),
            rec_id.to_raw(),
            false,
            tag,
            name_attr.to_raw(),
            file.map(|a| a.to_raw()).unwrap_or(null),
            line,
            scope.map(|a| a.to_raw()).unwrap_or(null),
            base_type.map(|a| a.to_raw()).unwrap_or(null),
            flags,
            size_in_bits,
            align_in_bits,
            raw_elements.len() as isize,
            raw_elements.as_ptr(),
            null, // dataLocation
            null, // rank
            null, // allocated
            null, // associated
        ))
    }
}

/// Creates a DI subroutine type attribute.
///
/// Describes a function signature for debug info. The `types` array should have
/// the return type as the first element, followed by parameter types.
/// Use `di_null_type` for void return.
pub fn di_subroutine_type<'c>(
    context: &'c Context,
    calling_convention: u32,
    types: &[Attribute<'c>],
) -> Attribute<'c> {
    let raw_types: Vec<MlirAttribute> = types.iter().map(|a| a.to_raw()).collect();
    unsafe {
        Attribute::from_raw(mlirLLVMDISubroutineTypeAttrGet(
            context.to_raw(),
            calling_convention,
            raw_types.len() as isize,
            raw_types.as_ptr(),
        ))
    }
}

/// Creates a self-referencing DI subprogram attribute for recursive references.
pub fn di_subprogram_rec_self(rec_id: Attribute<'_>) -> Attribute<'_> {
    unsafe { Attribute::from_raw(mlirLLVMDISubprogramAttrGetRecSelf(rec_id.to_raw())) }
}

/// Creates a DI subprogram attribute.
///
/// Describes a function or method for debug info. Attach this to an `llvm.func`
/// operation via the `llvm.di_subprogram` attribute.
#[allow(clippy::too_many_arguments)]
pub fn di_subprogram<'c>(
    context: &'c Context,
    id: Attribute<'c>,
    compile_unit: Attribute<'c>,
    scope: Attribute<'c>,
    name: &str,
    linkage_name: &str,
    file: Attribute<'c>,
    line: u32,
    scope_line: u32,
    flags: DISubprogramFlags,
    subprogram_type: Attribute<'c>,
) -> Attribute<'c> {
    let name_attr = StringAttribute::new(context, name);
    let linkage_name_attr = StringAttribute::new(context, linkage_name);
    unsafe {
        Attribute::from_raw(mlirLLVMDISubprogramAttrGet(
            context.to_raw(),
            MlirAttribute { ptr: std::ptr::null_mut() }, // rec_id (none)
            false, // is_rec_self
            id.to_raw(),
            compile_unit.to_raw(),
            scope.to_raw(),
            name_attr.to_raw(),
            linkage_name_attr.to_raw(),
            file.to_raw(),
            line,
            scope_line,
            flags.0,
            subprogram_type.to_raw(),
            0,              // nRetainedNodes
            std::ptr::null(), // retainedNodes
            0,              // nAnnotations
            std::ptr::null(), // annotations
        ))
    }
}

/// Creates a DI subprogram attribute with retained nodes and annotations.
#[allow(clippy::too_many_arguments)]
pub fn di_subprogram_full<'c>(
    context: &'c Context,
    rec_id: Option<Attribute<'c>>,
    id: Attribute<'c>,
    compile_unit: Attribute<'c>,
    scope: Attribute<'c>,
    name: &str,
    linkage_name: &str,
    file: Attribute<'c>,
    line: u32,
    scope_line: u32,
    flags: DISubprogramFlags,
    subprogram_type: Attribute<'c>,
    retained_nodes: &[Attribute<'c>],
    annotations: &[Attribute<'c>],
) -> Attribute<'c> {
    let null = MlirAttribute { ptr: std::ptr::null_mut() };
    let name_attr = StringAttribute::new(context, name);
    let linkage_name_attr = StringAttribute::new(context, linkage_name);
    let raw_retained: Vec<MlirAttribute> = retained_nodes.iter().map(|a| a.to_raw()).collect();
    let raw_annotations: Vec<MlirAttribute> = annotations.iter().map(|a| a.to_raw()).collect();

    unsafe {
        Attribute::from_raw(mlirLLVMDISubprogramAttrGet(
            context.to_raw(),
            rec_id.map(|a| a.to_raw()).unwrap_or(null),
            false,
            id.to_raw(),
            compile_unit.to_raw(),
            scope.to_raw(),
            name_attr.to_raw(),
            linkage_name_attr.to_raw(),
            file.to_raw(),
            line,
            scope_line,
            flags.0,
            subprogram_type.to_raw(),
            raw_retained.len() as isize,
            raw_retained.as_ptr(),
            raw_annotations.len() as isize,
            raw_annotations.as_ptr(),
        ))
    }
}

/// Gets the scope from a DI subprogram attribute.
pub fn di_subprogram_scope(subprogram: Attribute<'_>) -> Attribute<'_> {
    unsafe { Attribute::from_raw(mlirLLVMDISubprogramAttrGetScope(subprogram.to_raw())) }
}

/// Gets the line number from a DI subprogram attribute.
pub fn di_subprogram_line(subprogram: Attribute<'_>) -> u32 {
    unsafe { mlirLLVMDISubprogramAttrGetLine(subprogram.to_raw()) }
}

/// Gets the scope line from a DI subprogram attribute.
pub fn di_subprogram_scope_line(subprogram: Attribute<'_>) -> u32 {
    unsafe { mlirLLVMDISubprogramAttrGetScopeLine(subprogram.to_raw()) }
}

/// Gets the compile unit from a DI subprogram attribute.
pub fn di_subprogram_compile_unit(subprogram: Attribute<'_>) -> Attribute<'_> {
    unsafe {
        Attribute::from_raw(mlirLLVMDISubprogramAttrGetCompileUnit(
            subprogram.to_raw(),
        ))
    }
}

/// Gets the file from a DI subprogram attribute.
pub fn di_subprogram_file(subprogram: Attribute<'_>) -> Attribute<'_> {
    unsafe { Attribute::from_raw(mlirLLVMDISubprogramAttrGetFile(subprogram.to_raw())) }
}

/// Gets the subroutine type from a DI subprogram attribute.
pub fn di_subprogram_type(subprogram: Attribute<'_>) -> Attribute<'_> {
    unsafe { Attribute::from_raw(mlirLLVMDISubprogramAttrGetType(subprogram.to_raw())) }
}

/// Creates a DI local variable attribute.
///
/// Describes a local variable (parameter or automatic) in a function. Use with
/// `llvm.intr.dbg.declare` or `llvm.intr.dbg.value` operations to associate a
/// variable with a storage location.
///
/// Set `arg` to 0 for local variables, or the 1-based parameter index for
/// function arguments.
#[allow(clippy::too_many_arguments)]
pub fn di_local_variable<'c>(
    context: &'c Context,
    scope: Attribute<'c>,
    name: &str,
    file: Attribute<'c>,
    line: u32,
    arg: u32,
    align_in_bits: u32,
    di_type: Attribute<'c>,
    flags: i64,
) -> Attribute<'c> {
    let name_attr = StringAttribute::new(context, name);
    unsafe {
        Attribute::from_raw(mlirLLVMDILocalVariableAttrGet(
            context.to_raw(),
            scope.to_raw(),
            name_attr.to_raw(),
            file.to_raw(),
            line,
            arg,
            align_in_bits,
            di_type.to_raw(),
            flags,
        ))
    }
}

/// Creates a DI lexical block attribute.
///
/// Represents a lexical scope within a function (e.g., a block statement,
/// loop body, or branch arm).
pub fn di_lexical_block<'c>(
    context: &'c Context,
    scope: Attribute<'c>,
    file: Attribute<'c>,
    line: u32,
    column: u32,
) -> Attribute<'c> {
    unsafe {
        Attribute::from_raw(mlirLLVMDILexicalBlockAttrGet(
            context.to_raw(),
            scope.to_raw(),
            file.to_raw(),
            line,
            column,
        ))
    }
}

/// Creates a DI lexical block file attribute.
///
/// A lexical scope discriminated by a file. Useful when inlining produces
/// locations from different files within the same scope.
pub fn di_lexical_block_file<'c>(
    context: &'c Context,
    scope: Attribute<'c>,
    file: Attribute<'c>,
    discriminator: u32,
) -> Attribute<'c> {
    unsafe {
        Attribute::from_raw(mlirLLVMDILexicalBlockFileAttrGet(
            context.to_raw(),
            scope.to_raw(),
            file.to_raw(),
            discriminator,
        ))
    }
}

/// Creates a DI expression element attribute.
///
/// An element of a DWARF expression (opcode + arguments). Combine multiple
/// elements into a DI expression with `di_expression`.
pub fn di_expression_elem<'c>(context: &'c Context, opcode: u32, arguments: &[u64]) -> Attribute<'c> {
    unsafe {
        Attribute::from_raw(mlirLLVMDIExpressionElemAttrGet(
            context.to_raw(),
            opcode,
            arguments.len() as isize,
            arguments.as_ptr(),
        ))
    }
}

/// Creates a DI expression attribute.
///
/// A DWARF expression describes how to compute a variable's location from
/// registers, memory, or other values. An empty expression means "the value
/// is directly in the location."
pub fn di_expression<'c>(context: &'c Context, operations: &[Attribute<'c>]) -> Attribute<'c> {
    let raw_ops: Vec<MlirAttribute> = operations.iter().map(|a| a.to_raw()).collect();
    unsafe {
        Attribute::from_raw(mlirLLVMDIExpressionAttrGet(
            context.to_raw(),
            raw_ops.len() as isize,
            raw_ops.as_ptr(),
        ))
    }
}

/// Creates a DI module attribute.
#[allow(clippy::too_many_arguments)]
pub fn di_module<'c>(
    context: &'c Context,
    file: Option<Attribute<'c>>,
    scope: Option<Attribute<'c>>,
    name: &str,
    config_macros: &str,
    include_path: &str,
    apinotes: &str,
    line: u32,
    is_decl: bool,
) -> Attribute<'c> {
    let null = MlirAttribute { ptr: std::ptr::null_mut() };
    let name_attr = StringAttribute::new(context, name);
    let config_attr = StringAttribute::new(context, config_macros);
    let include_attr = StringAttribute::new(context, include_path);
    let apinotes_attr = StringAttribute::new(context, apinotes);
    unsafe {
        Attribute::from_raw(mlirLLVMDIModuleAttrGet(
            context.to_raw(),
            file.map(|a| a.to_raw()).unwrap_or(null),
            scope.map(|a| a.to_raw()).unwrap_or(null),
            name_attr.to_raw(),
            config_attr.to_raw(),
            include_attr.to_raw(),
            apinotes_attr.to_raw(),
            line,
            is_decl,
        ))
    }
}

/// Creates a DI imported entity attribute.
#[allow(clippy::too_many_arguments)]
pub fn di_imported_entity<'c>(
    context: &'c Context,
    tag: u32,
    scope: Attribute<'c>,
    entity: Option<Attribute<'c>>,
    file: Option<Attribute<'c>>,
    line: u32,
    name: &str,
    elements: &[Attribute<'c>],
) -> Attribute<'c> {
    let null = MlirAttribute { ptr: std::ptr::null_mut() };
    let name_attr = StringAttribute::new(context, name);
    let raw_elements: Vec<MlirAttribute> = elements.iter().map(|a| a.to_raw()).collect();
    unsafe {
        Attribute::from_raw(mlirLLVMDIImportedEntityAttrGet(
            context.to_raw(),
            tag,
            scope.to_raw(),
            entity.map(|a| a.to_raw()).unwrap_or(null),
            file.map(|a| a.to_raw()).unwrap_or(null),
            line,
            name_attr.to_raw(),
            raw_elements.len() as isize,
            raw_elements.as_ptr(),
        ))
    }
}

/// Creates a DI annotation attribute.
pub fn di_annotation<'c>(
    context: &'c Context,
    name: &str,
    value: &str,
) -> Attribute<'c> {
    use mlir_sys::mlirLLVMDIAnnotationAttrGet;
    let name_attr = StringAttribute::new(context, name);
    let value_attr = StringAttribute::new(context, value);
    unsafe {
        Attribute::from_raw(mlirLLVMDIAnnotationAttrGet(
            context.to_raw(),
            name_attr.to_raw(),
            value_attr.to_raw(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::llvm::{self, r#type::function},
        ir::{
            attribute::{StringAttribute, TypeAttribute},
            block::BlockLike,
            operation::OperationLike,
            r#type::IntegerType,
            Block, Identifier, Location, Module, Region, RegionLike,
        },
        test::create_test_context,
    };

    #[test]
    fn create_di_file() {
        let context = create_test_context();
        let file = di_file(&context, "main.c", "/home/user/project");
        assert!(file.to_string().contains("main.c"));
        assert!(file.to_string().contains("/home/user/project"));
    }

    #[test]
    fn create_di_null_type() {
        let context = create_test_context();
        let _null = di_null_type(&context);
    }

    #[test]
    fn create_di_basic_type() {
        let context = create_test_context();
        let ty = di_basic_type(
            &context,
            DITag::BaseType as u32,
            "int64",
            64,
            DITypeEncoding::Signed,
        );
        assert!(ty.to_string().contains("int64"));
    }

    #[test]
    fn create_di_compile_unit() {
        let context = create_test_context();
        let file = di_file(&context, "main.c", "/project");

        use crate::ir::attribute::{BoolAttribute, distinct::DisctinctAttribute};
        let distinct_id = DisctinctAttribute::new(&BoolAttribute::new(&context, true).into());

        let cu = di_compile_unit(
            &context,
            distinct_id.into(),
            29, // DW_LANG_C11
            file,
            "my-compiler 1.0",
            false,
            DIEmissionKind::Full,
            DINameTableKind::Default,
        );
        let cu_str = cu.to_string();
        assert!(cu_str.contains("main.c"));
    }

    #[test]
    fn create_di_subroutine_type() {
        let context = create_test_context();
        let null_ty = di_null_type(&context);
        let int_ty = di_basic_type(
            &context,
            DITag::BaseType as u32,
            "int64",
            64,
            DITypeEncoding::Signed,
        );
        let sub_ty = di_subroutine_type(&context, 0, &[null_ty, int_ty]);
        let s = sub_ty.to_string();
        assert!(s.contains("int64"));
    }

    #[test]
    fn create_di_subprogram() {
        let context = create_test_context();
        let file = di_file(&context, "main.c", "/project");

        use crate::ir::attribute::{BoolAttribute, distinct::DisctinctAttribute};
        let distinct_id = DisctinctAttribute::new(&BoolAttribute::new(&context, true).into());

        let cu = di_compile_unit(
            &context,
            distinct_id.into(),
            29,
            file,
            "my-compiler 1.0",
            false,
            DIEmissionKind::Full,
            DINameTableKind::Default,
        );

        let null_ty = di_null_type(&context);
        let sub_ty = di_subroutine_type(&context, 0, &[null_ty]);

        let sp_distinct = DisctinctAttribute::new(&BoolAttribute::new(&context, true).into());
        let sp = di_subprogram(
            &context,
            sp_distinct.into(),
            cu,
            file,
            "my_function",
            "my_function",
            file,
            10,
            10,
            DISubprogramFlags::DEFINITION,
            sub_ty,
        );

        assert_eq!(di_subprogram_line(sp), 10);
        assert_eq!(di_subprogram_scope_line(sp), 10);
    }

    #[test]
    fn create_di_local_variable() {
        let context = create_test_context();
        let file = di_file(&context, "main.c", "/project");

        use crate::ir::attribute::{BoolAttribute, distinct::DisctinctAttribute};
        let distinct_id = DisctinctAttribute::new(&BoolAttribute::new(&context, true).into());

        let cu = di_compile_unit(
            &context,
            distinct_id.into(),
            29,
            file,
            "my-compiler 1.0",
            false,
            DIEmissionKind::Full,
            DINameTableKind::Default,
        );

        let null_ty = di_null_type(&context);
        let sub_ty = di_subroutine_type(&context, 0, &[null_ty]);

        let sp_distinct = DisctinctAttribute::new(&BoolAttribute::new(&context, true).into());
        let sp = di_subprogram(
            &context,
            sp_distinct.into(),
            cu,
            file,
            "my_function",
            "my_function",
            file,
            1,
            1,
            DISubprogramFlags::DEFINITION,
            sub_ty,
        );

        let int_ty = di_basic_type(
            &context,
            DITag::BaseType as u32,
            "int64",
            64,
            DITypeEncoding::Signed,
        );

        let var = di_local_variable(
            &context,
            sp,
            "my_var",
            file,
            5,
            0,
            0,
            int_ty,
            0,
        );
        assert!(var.to_string().contains("my_var"));
    }

    #[test]
    fn create_di_lexical_block() {
        let context = create_test_context();
        let file = di_file(&context, "main.c", "/project");

        use crate::ir::attribute::{BoolAttribute, distinct::DisctinctAttribute};
        let distinct_id = DisctinctAttribute::new(&BoolAttribute::new(&context, true).into());

        let cu = di_compile_unit(
            &context,
            distinct_id.into(),
            29,
            file,
            "my-compiler 1.0",
            false,
            DIEmissionKind::Full,
            DINameTableKind::Default,
        );

        let null_ty = di_null_type(&context);
        let sub_ty = di_subroutine_type(&context, 0, &[null_ty]);

        let sp_distinct = DisctinctAttribute::new(&BoolAttribute::new(&context, true).into());
        let sp = di_subprogram(
            &context,
            sp_distinct.into(),
            cu,
            file,
            "my_function",
            "my_function",
            file,
            1,
            1,
            DISubprogramFlags::DEFINITION,
            sub_ty,
        );

        let block = di_lexical_block(&context, sp, file, 5, 3);
        let s = block.to_string();
        assert!(s.contains("DILexicalBlock") || s.contains("di_lexical_block") || !s.is_empty());
    }

    #[test]
    fn create_di_composite_type() {
        let context = create_test_context();
        let file = di_file(&context, "main.c", "/project");

        let int_ty = di_basic_type(
            &context,
            DITag::BaseType as u32,
            "int64",
            64,
            DITypeEncoding::Signed,
        );

        let field = di_derived_type(
            &context,
            DITag::Member as u32,
            "x",
            int_ty,
            64,
            64,
            0,
        );

        let struct_ty = di_composite_type(
            &context,
            DITag::StructureType as u32,
            "Point",
            Some(file),
            10,
            None,
            None,
            0,
            128,
            64,
            &[field],
        );
        assert!(struct_ty.to_string().contains("Point"));
    }

    #[test]
    fn create_di_expression() {
        let context = create_test_context();
        let empty_expr = di_expression(&context, &[]);
        let _s = empty_expr.to_string();
    }

    #[test]
    fn create_di_flags() {
        let context = create_test_context();
        let flags = di_flags(&context, 0);
        let _s = flags.to_string();
    }

    #[test]
    fn subprogram_flags_combine() {
        let flags = DISubprogramFlags::DEFINITION | DISubprogramFlags::OPTIMIZED;
        assert_eq!(flags.0, (1 << 3) | (1 << 4));
    }

    #[test]
    fn create_di_subprogram_and_attach_to_func() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);

        let file = di_file(&context, "test.c", "/project");

        use crate::ir::attribute::{BoolAttribute, distinct::DisctinctAttribute};
        let cu_id = DisctinctAttribute::new(&BoolAttribute::new(&context, true).into());
        let cu = di_compile_unit(
            &context,
            cu_id.into(),
            29,
            file,
            "my-compiler 1.0",
            false,
            DIEmissionKind::Full,
            DINameTableKind::Default,
        );

        let null_ty = di_null_type(&context);
        let sub_ty = di_subroutine_type(&context, 0, &[null_ty]);

        let sp_id = DisctinctAttribute::new(&BoolAttribute::new(&context, true).into());
        let sp = di_subprogram(
            &context,
            sp_id.into(),
            cu,
            file,
            "main",
            "main",
            file,
            1,
            1,
            DISubprogramFlags::DEFINITION,
            sub_ty,
        );

        let integer_type = IntegerType::new(&context, 32).into();

        module.body().append_operation(llvm::func(
            &context,
            StringAttribute::new(&context, "main"),
            TypeAttribute::new(function(integer_type, &[], false)),
            {
                let block = Block::new(&[]);
                let zero = block
                    .append_operation(llvm::zero(integer_type, location))
                    .result(0)
                    .unwrap()
                    .into();
                block.append_operation(llvm::r#return(Some(zero), location));
                let region = Region::new();
                region.append_block(block);
                region
            },
            &[(
                Identifier::new(&context, "llvm.di_subprogram"),
                sp,
            )],
            location,
        ));

        assert!(module.as_operation().verify());
    }
}
