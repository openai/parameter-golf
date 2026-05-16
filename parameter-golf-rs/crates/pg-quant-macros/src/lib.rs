use proc_macro::{Delimiter, Literal, TokenStream, TokenTree};
use std::collections::BTreeMap;

#[proc_macro]
pub fn quant_layout(input: TokenStream) -> TokenStream {
    match compile_quant_layout(input) {
        Ok(tokens) => tokens,
        Err(message) => compile_error(&message),
    }
}

fn compile_quant_layout(input: TokenStream) -> Result<TokenStream, String> {
    let tokens = unwrap_single_group(input);
    let fields = parse_fields(tokens)?;
    let name = required_string(&fields, "name")?;
    let matrix_bits = required_u8(&fields, "matrix_bits")?;
    let mlp_bits = required_u8(&fields, "mlp_bits")?;
    let embed_bits = required_u8(&fields, "embed_bits")?;
    let attn_gate_bits = required_u8(&fields, "attn_gate_bits")?;
    let gptq_calibration_batches = required_usize(&fields, "gptq_calibration_batches")?;
    let target_artifact_bytes = required_usize(&fields, "target_artifact_bytes")?;
    let lqer_enabled = optional_bool(&fields, "lqer_enabled", false)?;
    reject_unknown_fields(
        &fields,
        &[
            "name",
            "matrix_bits",
            "mlp_bits",
            "embed_bits",
            "attn_gate_bits",
            "gptq_calibration_batches",
            "target_artifact_bytes",
            "lqer_enabled",
        ],
    )?;
    for (field, bits) in [
        ("matrix_bits", matrix_bits),
        ("mlp_bits", mlp_bits),
        ("embed_bits", embed_bits),
        ("attn_gate_bits", attn_gate_bits),
    ] {
        if !(2..=8).contains(&bits) {
            return Err(format!(
                "{field}={bits} is outside the supported 2..=8 range"
            ));
        }
    }
    let output = format!(
        "CompiledQuantLayout {{ name: {name:?}, matrix_bits: {matrix_bits}, mlp_bits: {mlp_bits}, embed_bits: {embed_bits}, attn_gate_bits: {attn_gate_bits}, gptq_calibration_batches: {gptq_calibration_batches}, target_artifact_bytes: {target_artifact_bytes}, lqer_enabled: {lqer_enabled} }}"
    );
    output
        .parse()
        .map_err(|err| format!("internal quant_layout expansion failed: {err}"))
}

fn reject_unknown_fields(
    fields: &BTreeMap<String, TokenTree>,
    allowed: &[&str],
) -> Result<(), String> {
    for key in fields.keys() {
        if !allowed.iter().any(|allowed_key| allowed_key == key) {
            return Err(format!("unknown quant_layout field `{key}`"));
        }
    }
    Ok(())
}

fn unwrap_single_group(input: TokenStream) -> Vec<TokenTree> {
    let tokens: Vec<TokenTree> = input.into_iter().collect();
    if let [TokenTree::Group(group)] = tokens.as_slice() {
        if group.delimiter() == Delimiter::Brace {
            return group.stream().into_iter().collect();
        }
    }
    tokens
}

fn parse_fields(tokens: Vec<TokenTree>) -> Result<BTreeMap<String, TokenTree>, String> {
    let mut fields = BTreeMap::new();
    let mut iter = tokens.into_iter().peekable();
    while let Some(token) = iter.next() {
        let key = match token {
            TokenTree::Ident(ident) => ident.to_string(),
            TokenTree::Punct(punct) if punct.as_char() == ',' => continue,
            other => return Err(format!("expected field name, got `{other}`")),
        };
        match iter.next() {
            Some(TokenTree::Punct(punct)) if punct.as_char() == ':' => {}
            Some(other) => return Err(format!("expected `:` after `{key}`, got `{other}`")),
            None => return Err(format!("expected `:` after `{key}`")),
        }
        let value = iter
            .next()
            .ok_or_else(|| format!("expected value after `{key}:`"))?;
        if fields.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate quant_layout field `{key}`"));
        }
        if matches!(iter.peek(), Some(TokenTree::Punct(punct)) if punct.as_char() == ',') {
            iter.next();
        }
    }
    Ok(fields)
}

fn required_string(fields: &BTreeMap<String, TokenTree>, key: &str) -> Result<String, String> {
    let token = fields
        .get(key)
        .ok_or_else(|| format!("missing required quant_layout field `{key}`"))?;
    match token {
        TokenTree::Literal(lit) => literal_string(lit)
            .ok_or_else(|| format!("quant_layout field `{key}` must be a string literal")),
        other => Err(format!(
            "quant_layout field `{key}` must be a string literal, got `{other}`"
        )),
    }
}

fn required_u8(fields: &BTreeMap<String, TokenTree>, key: &str) -> Result<u8, String> {
    let token = fields
        .get(key)
        .ok_or_else(|| format!("missing required quant_layout field `{key}`"))?;
    match token {
        TokenTree::Literal(lit) => lit
            .to_string()
            .parse::<u8>()
            .map_err(|_| format!("quant_layout field `{key}` must be an integer literal")),
        other => Err(format!(
            "quant_layout field `{key}` must be an integer literal, got `{other}`"
        )),
    }
}

fn required_usize(fields: &BTreeMap<String, TokenTree>, key: &str) -> Result<usize, String> {
    let token = fields
        .get(key)
        .ok_or_else(|| format!("missing required quant_layout field `{key}`"))?;
    match token {
        TokenTree::Literal(lit) => lit
            .to_string()
            .replace('_', "")
            .parse::<usize>()
            .map_err(|_| format!("quant_layout field `{key}` must be an integer literal")),
        other => Err(format!(
            "quant_layout field `{key}` must be an integer literal, got `{other}`"
        )),
    }
}

fn optional_bool(
    fields: &BTreeMap<String, TokenTree>,
    key: &str,
    default_value: bool,
) -> Result<bool, String> {
    let Some(token) = fields.get(key) else {
        return Ok(default_value);
    };
    match token {
        TokenTree::Ident(ident) if ident.to_string() == "true" => Ok(true),
        TokenTree::Ident(ident) if ident.to_string() == "false" => Ok(false),
        other => Err(format!(
            "quant_layout field `{key}` must be true or false, got `{other}`"
        )),
    }
}

fn literal_string(lit: &Literal) -> Option<String> {
    let raw = lit.to_string();
    raw.strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .map(|s| s.replace("\\\"", "\""))
}

fn compile_error(message: &str) -> TokenStream {
    format!("compile_error!({message:?});")
        .parse()
        .expect("compile_error expansion must parse")
}
