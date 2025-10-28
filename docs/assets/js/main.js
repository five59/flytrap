// Flytrap Documentation JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Add copy-to-clipboard functionality to code blocks
    addCopyButtons();

    // Add smooth scrolling to anchor links
    addSmoothScrolling();

    // Add table of contents generation
    generateTableOfContents();

    // Highlight current page in navigation
    highlightCurrentPage();

    // Add responsive navigation toggle
    addMobileNavigation();
});

function addCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre');

    codeBlocks.forEach(function(block) {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: #3498db;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        `;

        const wrapper = document.createElement('div');
        wrapper.style.position = 'relative';
        block.parentNode.insertBefore(wrapper, block);
        wrapper.appendChild(block);
        wrapper.appendChild(button);

        button.addEventListener('click', function() {
            const code = block.textContent || block.innerText;
            navigator.clipboard.writeText(code).then(function() {
                button.textContent = 'Copied!';
                button.style.background = '#27ae60';
                setTimeout(function() {
                    button.textContent = 'Copy';
                    button.style.background = '#3498db';
                }, 2000);
            });
        });
    });
}

function addSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');

    links.forEach(function(link) {
        link.addEventListener('click', function(e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

function generateTableOfContents() {
    const content = document.querySelector('.content');
    if (!content) return;

    const headings = content.querySelectorAll('h2, h3');
    if (headings.length < 3) return; // Only generate TOC if there are enough headings

    const toc = document.createElement('nav');
    toc.className = 'table-of-contents';
    toc.innerHTML = '<h3>Table of Contents</h3><ul></ul>';

    const ul = toc.querySelector('ul');

    headings.forEach(function(heading) {
        const li = document.createElement('li');
        const a = document.createElement('a');
        const id = heading.textContent.toLowerCase().replace(/[^a-z0-9]+/g, '-');

        heading.id = id;
        a.href = '#' + id;
        a.textContent = heading.textContent;
        a.style.paddingLeft = heading.tagName === 'H3' ? '20px' : '0';

        li.appendChild(a);
        ul.appendChild(li);
    });

    // Insert TOC after the main heading
    const mainHeading = content.querySelector('h1');
    if (mainHeading) {
        mainHeading.insertAdjacentElement('afterend', toc);
    }

    // Add TOC styles
    const style = document.createElement('style');
    style.textContent = `
        .table-of-contents {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .table-of-contents h3 {
            margin-bottom: 0.5rem;
            color: #2c3e50;
        }
        .table-of-contents ul {
            list-style: none;
            padding: 0;
        }
        .table-of-contents li {
            margin-bottom: 0.25rem;
        }
        .table-of-contents a {
            color: #3498db;
            text-decoration: none;
        }
        .table-of-contents a:hover {
            text-decoration: underline;
        }
    `;
    document.head.appendChild(style);
}

function highlightCurrentPage() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-list a');

    navLinks.forEach(function(link) {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('current');
            link.style.backgroundColor = 'rgba(255,255,255,0.2)';
        }
    });
}

function addMobileNavigation() {
    const nav = document.querySelector('.site-nav');
    if (!nav) return;

    // Create mobile menu toggle
    const toggle = document.createElement('button');
    toggle.className = 'nav-toggle';
    toggle.innerHTML = '☰';
    toggle.style.cssText = `
        display: none;
        background: none;
        border: none;
        color: white;
        font-size: 1.5rem;
        cursor: pointer;
    `;

    nav.insertBefore(toggle, nav.querySelector('.nav-list'));

    // Add responsive styles
    const style = document.createElement('style');
    style.textContent = `
        @media (max-width: 768px) {
            .nav-toggle {
                display: block;
            }
            .nav-list {
                display: none;
                flex-direction: column;
                position: absolute;
                top: 100%;
                left: 0;
                right: 0;
                background: #2c3e50;
                padding: 1rem;
            }
            .nav-list.show {
                display: flex;
            }
        }
    `;
    document.head.appendChild(style);

    // Toggle navigation
    toggle.addEventListener('click', function() {
        const navList = document.querySelector('.nav-list');
        navList.classList.toggle('show');
    });
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K to focus search (if implemented)
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        // Focus search input if it exists
    }

    // Escape to close mobile menu
    if (e.key === 'Escape') {
        const navList = document.querySelector('.nav-list');
        if (navList && navList.classList.contains('show')) {
            navList.classList.remove('show');
        }
    }
});