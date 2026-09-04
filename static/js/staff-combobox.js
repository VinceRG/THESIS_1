function initStaffCombobox(root) {
  const searchInput = root.querySelector('[data-staff-search]');
  const valueInput = root.querySelector('[data-staff-value]');
  const list = root.querySelector('[data-staff-list]');
  const emptyState = root.querySelector('[data-staff-empty]');
  const options = Array.from(root.querySelectorAll('.staff-combobox-option'));
  const groupLabels = Array.from(root.querySelectorAll('.staff-combobox-group-label'));
  if (!searchInput || !valueInput || !list || !options.length) return;

  let activeIndex = -1;

  const visibleOptions = () => options.filter(option => option.style.display !== 'none');

  function closeList() {
    list.hidden = true;
    activeIndex = -1;
    options.forEach(option => option.classList.remove('is-active'));
  }

  function applyFilter(term) {
    const query = term.trim().toLowerCase();
    let anyVisible = false;
    options.forEach(option => {
      const matches = !query || option.dataset.search.includes(query);
      option.style.display = matches ? '' : 'none';
      if (matches) anyVisible = true;
    });
    groupLabels.forEach(label => {
      let sibling = label.nextElementSibling;
      let hasVisible = false;
      while (sibling && sibling.classList.contains('staff-combobox-option')) {
        if (sibling.style.display !== 'none') hasVisible = true;
        sibling = sibling.nextElementSibling;
      }
      label.style.display = hasVisible ? '' : 'none';
    });
    if (emptyState) emptyState.hidden = anyVisible;
    activeIndex = -1;
    options.forEach(option => option.classList.remove('is-active'));
  }

  function selectOption(option) {
    valueInput.value = option.dataset.value;
    searchInput.value = option.dataset.label;
    root.classList.remove('is-invalid');
    closeList();
  }

  function setActive(nextIndex) {
    const visible = visibleOptions();
    if (!visible.length) return;
    activeIndex = (nextIndex + visible.length) % visible.length;
    options.forEach(option => option.classList.remove('is-active'));
    const active = visible[activeIndex];
    active.classList.add('is-active');
    active.scrollIntoView({ block: 'nearest' });
  }

  searchInput.addEventListener('focus', () => {
    list.hidden = false;
    applyFilter('');
  });

  searchInput.addEventListener('blur', () => {
    closeList();
  });

  searchInput.addEventListener('input', () => {
    valueInput.value = '';
    list.hidden = false;
    applyFilter(searchInput.value);
  });

  searchInput.addEventListener('keydown', event => {
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      list.hidden = false;
      setActive(activeIndex + 1);
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      list.hidden = false;
      setActive(activeIndex - 1);
    } else if (event.key === 'Enter') {
      if (!list.hidden && activeIndex >= 0) {
        event.preventDefault();
        selectOption(visibleOptions()[activeIndex]);
      }
    } else if (event.key === 'Escape') {
      closeList();
    }
  });

  options.forEach(option => {
    option.addEventListener('mousedown', event => {
      event.preventDefault();
      selectOption(option);
    });
  });

  const form = root.closest('form');
  if (form) {
    form.addEventListener('submit', event => {
      if (!valueInput.value) {
        event.preventDefault();
        root.classList.add('is-invalid');
        searchInput.focus();
      }
    });
  }
}

document.querySelectorAll('[data-staff-combobox]').forEach(initStaffCombobox);
