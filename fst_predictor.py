"""
Enhanced Finite State Transducer for structural web text prediction.
OAG principle: structured knowledge augments neural prediction.

Design: patterns ranked by FREQUENCY in FineWeb × PREDICTABILITY.
High-frequency + high-predictability = biggest BPB impact.
"""
import torch
from typing import Optional, Tuple

class WebTextFST:
    # === HTML: ~12-15% of FineWeb bytes ===
    # Ranked by frequency in web crawls
    TAGS = [
        'div','p','span','a','li','ul','br','img','td','tr','th',
        'table','h1','h2','h3','h4','h5','h6','ol','dl','dt','dd',
        'form','input','button','label','select','option','textarea',
        'header','footer','nav','main','section','article','aside',
        'script','style','link','meta','title','head','body','html',
        'strong','em','b','i','u','pre','code','blockquote','figure',
        'figcaption','video','audio','source','canvas','svg','path',
        'iframe','noscript','small','sub','sup','abbr','cite',
    ]

    # Self-closing tags (no </tag> expected)
    VOID_TAGS = {'br','hr','img','input','meta','link','source','area','base','col','embed','track','wbr'}

    ATTRS = [
        'class','id','href','src','alt','title','style','type',
        'name','value','rel','width','height','target','action',
        'method','placeholder','role','aria-label','aria-hidden',
        'data-','content','charset','http-equiv','property',
        'sizes','media','loading','decoding','crossorigin',
    ]

    # === BOILERPLATE: ~3-5% of bytes, near-deterministic ===
    BOILERPLATE = sorted([
        # Legal/footer (extremely common)
        'All rights reserved',
        'Privacy Policy',
        'Terms of Service',
        'Terms and Conditions',
        'Terms of Use',
        'Cookie Policy',
        'Cookie Settings',
        'Acceptable Use Policy',
        # Navigation (every website)
        'Contact Us',
        'About Us',
        'About us',
        'Sign in',
        'Sign up',
        'Log in',
        'Log out',
        'Create account',
        'My account',
        'Shopping cart',
        'Checkout',
        'Home',
        'Search',
        'Menu',
        # CTAs (calls to action)
        'Subscribe to our newsletter',
        'Subscribe to our',
        'Sign up for our',
        'Follow us on',
        'Share this',
        'Read more',
        'Learn more',
        'Click here',
        'Download now',
        'Get started',
        'Try it free',
        'Buy now',
        'Add to cart',
        'View all',
        'Show more',
        'Load more',
        'See more',
        'Back to top',
        # Social media
        'Follow us on Twitter',
        'Follow us on Facebook',
        'Follow us on Instagram',
        'Follow us on LinkedIn',
        'Share on Twitter',
        'Share on Facebook',
        # Common page elements
        'Powered by',
        'Copyright',
        'All Rights Reserved',
        'Published on',
        'Last updated',
        'Written by',
        'Posted by',
        'Filed under',
        'Tagged with',
        'Comments',
        'Leave a comment',
        'Leave a reply',
        'Related posts',
        'Related articles',
        'You may also like',
        'Recommended for you',
        'Popular posts',
        'Recent posts',
        'Categories',
        'Archives',
        'Newsletter',
        'Subscribe',
        'Unsubscribe',
        # E-commerce
        'Free shipping',
        'Add to wishlist',
        'Out of stock',
        'In stock',
        'Customer reviews',
        'Product description',
        'Shipping information',
        'Return policy',
        # Cookie consent (on almost every EU site)
        'This website uses cookies',
        'We use cookies',
        'Accept all cookies',
        'Reject all',
        'Cookie preferences',
        'Manage preferences',
        'Necessary cookies',
        'By continuing to use',
        'By clicking',
    ], key=len, reverse=True)

    # === COMMON MULTI-WORD PHRASES (English, very high frequency) ===
    PHRASES = sorted([
        'in order to',
        'as well as',
        'such as',
        'due to',
        'in addition to',
        'according to',
        'in terms of',
        'on the other hand',
        'at the same time',
        'in the United States',
        'for more information',
        'for example',
        'in this case',
        'at least',
        'at the end of',
        'one of the',
        'as a result',
        'the United States',
        'the United Kingdom',
        'New York',
        'Los Angeles',
        'San Francisco',
    ], key=len, reverse=True)

    # === URL TLD patterns ===
    TLDS = ['.com','.org','.net','.edu','.gov','.io','.co.uk','.de','.fr','.ru','.cn','.info','.biz']

    # === Common file extensions ===
    EXTENSIONS = ['.html','.htm','.php','.asp','.aspx','.jsp','.css','.js','.json','.xml','.jpg','.png','.gif','.pdf','.svg']

    def __init__(self, tokenizer):
        self.sp = tokenizer
        self.vocab_size = tokenizer.vocab_size()
        self._cache = {}
        self._tag_stack = []  # Track open HTML tags for closing prediction

    def _to_probs(self, predicted_str: str, confidence: float) -> Optional[torch.Tensor]:
        key = (predicted_str, round(confidence, 2))
        if key in self._cache:
            return self._cache[key].clone()
        tokens = self.sp.encode(predicted_str)
        if not tokens:
            return None
        probs = torch.full((self.vocab_size,), (1.0 - confidence) / self.vocab_size)
        probs[tokens[0]] = confidence
        probs = probs / probs.sum()
        self._cache[key] = probs.clone()
        return probs

    def predict(self, context_text: str) -> Tuple[Optional[torch.Tensor], float]:
        if len(context_text) < 2:
            return None, 0.0
        ctx = context_text[-300:]  # Larger window for better matching

        # =============================================
        # HTML PATTERNS (highest impact — ~12% of bytes)
        # =============================================

        # Closing tag: after "</" predict tag name
        if ctx.endswith('</'):
            # Search for most recent unclosed open tag
            for tag in self.TAGS:
                open_pattern = f'<{tag}'
                close_pattern = f'</{tag}'
                # Count opens vs closes
                opens = ctx.count(open_pattern)
                closes = ctx.count(close_pattern)
                if opens > closes and tag not in self.VOID_TAGS:
                    return self._to_probs(tag, 0.80), 0.80
            # Fallback: predict 'div' (most common)
            return self._to_probs('div', 0.35), 0.35

        # After ">" likely starts content or another tag
        if ctx.endswith('>\n') or ctx.endswith('>\r\n'):
            return self._to_probs('<', 0.30), 0.30

        # Inside tag after space: predict attribute
        last_lt = ctx.rfind('<')
        last_gt = ctx.rfind('>')
        if last_lt > last_gt:
            tag_content = ctx[last_lt:]
            # After tag name + space: attribute
            if tag_content.endswith(' '):
                # Predict most likely attribute based on tag
                if '<a ' in tag_content:
                    return self._to_probs('href="', 0.65), 0.65
                elif '<img ' in tag_content:
                    return self._to_probs('src="', 0.55), 0.55
                elif '<link ' in tag_content:
                    return self._to_probs('rel="', 0.50), 0.50
                elif '<meta ' in tag_content:
                    return self._to_probs('content="', 0.40), 0.40
                elif '<input ' in tag_content:
                    return self._to_probs('type="', 0.50), 0.50
                else:
                    return self._to_probs('class="', 0.45), 0.45

            # After attribute="value" expect space or >
            if tag_content.endswith('"'):
                # Count quotes to see if we just closed a value
                quotes = tag_content.count('"')
                if quotes % 2 == 0 and quotes >= 2:
                    return self._to_probs(' ', 0.40), 0.40

        # =============================================
        # URL PATTERNS (high impact — ~8% of bytes)
        # =============================================

        if ctx.endswith('http'):
            return self._to_probs('s://', 0.90), 0.90
        if ctx.endswith('https:'):
            return self._to_probs('//', 0.97), 0.97
        if ctx.endswith('http:'):
            return self._to_probs('//', 0.95), 0.95
        if ctx.endswith('https://') or ctx.endswith('http://'):
            return self._to_probs('www.', 0.50), 0.50

        # TLD prediction after domain
        for tld in self.TLDS:
            prefix = tld[:-1]  # e.g., '.co' for '.com'
            if len(prefix) >= 2 and ctx.endswith(prefix):
                remaining = tld[len(prefix):]
                return self._to_probs(remaining, 0.60), 0.60

        # File extension prediction
        for ext in self.EXTENSIONS:
            prefix = ext[:-1]
            if len(prefix) >= 3 and ctx.endswith(prefix):
                remaining = ext[len(prefix):]
                return self._to_probs(remaining, 0.55), 0.55

        # =============================================
        # JSON PATTERNS (~3-5% of bytes)
        # =============================================

        stripped = ctx.rstrip()
        if stripped.endswith('{'):
            return self._to_probs('"', 0.80), 0.80
        if stripped.endswith('['):
            return self._to_probs('"', 0.45), 0.45
        # After "key": predict value start
        if stripped.endswith('":'):
            return self._to_probs(' "', 0.40), 0.40
        if stripped.endswith('": '):
            return self._to_probs('"', 0.50), 0.50
        # After value, predict comma or closing
        if stripped.endswith('",'):
            return self._to_probs(' "', 0.45), 0.45

        # =============================================
        # DATE/NUMBER PATTERNS (~2-3% of bytes)
        # =============================================

        if ctx.endswith('202'):
            return self._to_probs('5', 0.35), 0.35  # 2025 slightly more common than 2026
        if ctx.endswith('2025-') or ctx.endswith('2026-') or ctx.endswith('2024-'):
            return self._to_probs('0', 0.30), 0.30  # Month starts with 0 or 1
        if ctx.endswith('January') or ctx.endswith('February') or ctx.endswith('March'):
            return self._to_probs(' ', 0.85), 0.85
        if ctx.endswith('Monday') or ctx.endswith('Tuesday') or ctx.endswith('Wednesday'):
            return self._to_probs(',', 0.60), 0.60

        # =============================================
        # BOILERPLATE (~3-5% of bytes, very predictable)
        # =============================================

        for phrase in self.BOILERPLATE:
            for start_pos in range(max(0, len(ctx)-len(phrase)), len(ctx)):
                prefix = ctx[start_pos:]
                if phrase.startswith(prefix) and 4 <= len(prefix) < len(phrase):
                    continuation = phrase[len(prefix):len(prefix)+5]
                    # Confidence scales with match length
                    conf = min(0.92, 0.4 + len(prefix) * 0.04)
                    return self._to_probs(continuation, conf), conf

        # =============================================
        # COMMON PHRASES (~5% of bytes)
        # =============================================

        for phrase in self.PHRASES:
            for start_pos in range(max(0, len(ctx)-len(phrase)), len(ctx)):
                prefix = ctx[start_pos:]
                if phrase.startswith(prefix) and 4 <= len(prefix) < len(phrase):
                    continuation = phrase[len(prefix):len(prefix)+4]
                    conf = min(0.75, 0.3 + len(prefix) * 0.03)
                    return self._to_probs(continuation, conf), conf

        # =============================================
        # PUNCTUATION PATTERNS (high frequency)
        # =============================================

        # After sentence-ending period + space, predict capital letter
        if ctx.endswith('. '):
            return self._to_probs('The', 0.08), 0.08  # Low conf, just a bias

        # After comma + space in a list
        if ctx.endswith(', and '):
            return self._to_probs('the', 0.10), 0.10

        return None, 0.0
